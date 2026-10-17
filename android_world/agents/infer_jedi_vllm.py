"""Jedi-7B-1080p wrapper that talks to a vLLM OpenAI-compatible server.

Jedi is a Qwen2.5-VL-7B-Instruct fine-tune from xlangai for
high-resolution GUI grounding. The HF transformers in-process path
also works (clean Qwen2.5-VL architecture, no custom modeling code),
but vLLM is the recommended deployment for sharing weights across
multiple emulators via continuous batching.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Optional

import numpy as np

from android_world.agents import infer

ERROR_CALLING_LLM = infer.ERROR_CALLING_LLM


class JediVLLMWrapper(infer.LlmWrapper, infer.MultimodalLlmWrapper):
  """vLLM-backed Jedi-7B-1080p wrapper.

  Jedi's native action schema is the Anthropic Computer-Use tool call:

      <tool_call>
      {"name": "computer", "arguments": {"action": "left_click",
                                          "coordinate": [x, y]}}
      </tool_call>

  Empirical: M3A's "Reason:/Action: <JSON>" prompt makes Jedi emit
  free-form English descriptions; the pyautogui prompt borrowed from
  OpenCUA also doesn't fit. Sending the native tool_call recipe
  (mirrored from the user prompt template) is what makes Jedi produce
  parseable output.
  """

  _SYSTEM_PROMPT = (
      "You are a GUI agent operating a touchscreen. Perform tool calls "
      "using the `computer` tool to complete the user's task. Emit each "
      "action inside <tool_call>...</tool_call> tags as JSON of the "
      'form {"name": "computer", "arguments": {"action": "left_click", '
      '"coordinate": [x, y]}}. Do not write prose outside the tags.'
  )

  def __init__(
      self,
      base_url: str = "http://127.0.0.1:4245/v1",
      model_name: str = "jedi-7b-1080p",
      api_key: str = "EMPTY",
      max_new_tokens: int = 512,
      temperature: float = 0.0,
      timeout: float = 180.0,
      # Coord-scaling: server-side smart_resize uses these caps; the
      # agent loop reads them to map model coords back to native pixels.
      min_pixels: int = 3136,
      max_pixels: int = 2007040,
  ):
    try:
      from openai import OpenAI  # pylint: disable=import-outside-toplevel
    except ImportError as e:
      raise ImportError(
          "openai package required for JediVLLMWrapper. "
          "Install via `pip install 'openai>=1.0'`."
      ) from e

    self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    self.base_url = base_url
    self.model_name = model_name
    self.max_new_tokens = max_new_tokens
    self.temperature = temperature
    self.min_pixels = min_pixels
    self.max_pixels = max_pixels
    self._call_no = 0

    try:
      models = self._client.models.list()
      served = [m.id for m in (models.data or [])]
      print(
          f"[Jedi-vLLM] connected {base_url} | served={served}",
          flush=True,
      )
      if model_name not in served:
        print(
            f"[Jedi-vLLM] WARNING: '{model_name}' not in {served}. "
            f"Override --served-model-name on the server or pass a "
            f"matching value to the wrapper.",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
      raise RuntimeError(
          f"vLLM server unreachable at {base_url}: {type(e).__name__}: {e}\n"
          f"Start it first: scripts/start_jedi_vllm.sh"
      ) from e

  # ------------------------------------------------------------------ interface

  def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    try:
      from PIL import Image  # pylint: disable=import-outside-toplevel
    except ImportError as e:
      raise ImportError("Pillow required for JediVLLMWrapper.") from e

    self._call_no += 1
    call_no = self._call_no
    t0 = time.time()
    print(
        f"[Jedi-vLLM #{call_no}] predict_mm start  (images={len(images)})",
        flush=True,
    )

    try:
      content: list[dict] = []
      for img in images:
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
      content.append({"type": "text", "text": text_prompt})

      messages = [
          {"role": "system", "content": self._SYSTEM_PROMPT},
          {"role": "user", "content": content},
      ]

      response = self._client.chat.completions.create(
          model=self.model_name,
          messages=messages,
          temperature=self.temperature,
          max_tokens=self.max_new_tokens,
      )

      output_text = response.choices[0].message.content or ""
      usage = response.usage
      in_tokens = int(getattr(usage, "prompt_tokens", 0)) if usage else 0
      out_tokens = int(getattr(usage, "completion_tokens", 0)) if usage else 0
      elapsed = time.time() - t0
      preview = " ".join(output_text.split())[:140]
      print(
          f"[Jedi-vLLM #{call_no}] done in {elapsed:.1f}s  "
          f"in={in_tokens}  out={out_tokens}  | {preview}",
          flush=True,
      )
      return output_text, None, (output_text or "ok")

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[Jedi-vLLM #{call_no}] FAILED after {time.time()-t0:.1f}s: "
          f"{type(e).__name__}: {e}",
          flush=True,
      )
      logging.exception("[Jedi-vLLM] predict_mm failed")
      return ERROR_CALLING_LLM, None, None
