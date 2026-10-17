"""GUI-Owl-7B wrapper that talks to a vLLM OpenAI-compatible server.

GUI-Owl is a Qwen2.5-VL-7B-Instruct fine-tune for GUI automation
(Mobile-Agent-V3 project). The HF transformers in-process path works
fine — but the model card explicitly recommends vLLM for production
deployment, and serving makes it cheap to share weights across
multiple emulators (continuous batching).

Same `predict_mm` contract as `infer_qwen_vl.QwenVLWrapper`. The M3A
"Reason:/Action:" system prompt is verbatim from QwenVLWrapper since
GUI-Owl is a Qwen2.5-VL fine-tune that complies with the M3A format
(verified via smoke run).
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


class GUIOwlVLLMWrapper(infer.LlmWrapper, infer.MultimodalLlmWrapper):
  """vLLM-backed GUI-Owl-7B wrapper for M3A / T3A."""

  _SYSTEM_PROMPT = (
      "You are a precise Android Automotive GUI agent. "
      "When answering an action selection prompt you MUST output EXACTLY two lines, "
      "in English, with no markdown, no code fences, no preamble, no extra text:\n"
      "Reason: <one short sentence>\n"
      "Action: <one JSON object on a single line>\n"
      "Always emit the literal English words `Reason:` and `Action:` with this "
      "capitalization and ordering. The JSON must be valid and contain "
      "`action_type`. Do not wrap the JSON in backticks."
  )

  def __init__(
      self,
      base_url: str = "http://127.0.0.1:4244/v1",
      model_name: str = "gui-owl-7b",
      api_key: str = "EMPTY",
      max_new_tokens: int = 1024,
      temperature: float = 0.0,
      timeout: float = 180.0,
  ):
    try:
      from openai import OpenAI  # pylint: disable=import-outside-toplevel
    except ImportError as e:
      raise ImportError(
          "openai package required for GUIOwlVLLMWrapper. "
          "Install via `pip install 'openai>=1.0'`."
      ) from e

    self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    self.base_url = base_url
    self.model_name = model_name
    self.max_new_tokens = max_new_tokens
    self.temperature = temperature
    self._call_no = 0

    try:
      models = self._client.models.list()
      served = [m.id for m in (models.data or [])]
      print(
          f"[GUI-Owl-vLLM] connected {base_url} | served={served}",
          flush=True,
      )
      if model_name not in served:
        print(
            f"[GUI-Owl-vLLM] WARNING: '{model_name}' not in {served}. "
            f"Override --served-model-name on the server or pass a "
            f"matching value to the wrapper.",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
      raise RuntimeError(
          f"vLLM server unreachable at {base_url}: {type(e).__name__}: {e}\n"
          f"Start it first: scripts/start_gui_owl_vllm.sh"
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
      raise ImportError("Pillow required for GUIOwlVLLMWrapper.") from e

    self._call_no += 1
    call_no = self._call_no
    t0 = time.time()
    print(
        f"[GUI-Owl-vLLM #{call_no}] predict_mm start  (images={len(images)})",
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
          f"[GUI-Owl-vLLM #{call_no}] done in {elapsed:.1f}s  "
          f"in={in_tokens}  out={out_tokens}  | {preview}",
          flush=True,
      )
      return output_text, None, (output_text or "ok")

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[GUI-Owl-vLLM #{call_no}] FAILED after {time.time()-t0:.1f}s: "
          f"{type(e).__name__}: {e}",
          flush=True,
      )
      logging.exception("[GUI-Owl-vLLM] predict_mm failed")
      return ERROR_CALLING_LLM, None, None
