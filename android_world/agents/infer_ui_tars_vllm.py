"""UI-TARS-1.5-7B wrapper that talks to a vLLM OpenAI-compatible server.

Mirrors the in-process `infer_ui_tars.UITarsWrapper` predict_mm
contract (returns (text, None, raw)) but sends multimodal Chat
Completions over HTTP. The system prompt is intentionally minimal —
the per-step user prompt from `m3a_ui_tars_native._NATIVE_PROMPT_TEMPLATE`
is comprehensive (Thought + native click(point=...) action enumeration),
so we don't need to layer another instruction on top.

Use with `m3a_ui_tars_native_vllm.build_*` factory; the agent loop
already handles native coordinate parsing, smart-resize remap, and
stale-state warning.
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


class UITarsVLLMWrapper(infer.LlmWrapper, infer.MultimodalLlmWrapper):
  """vLLM-backed UI-TARS-1.5-7B wrapper."""

  # UI-TARS native is trained on the MOBILE_USE_DOUBAO / COMPUTER_USE_DOUBAO
  # template. The agent's per-step user prompt restates the action space
  # explicitly, so we keep this short to avoid drift.
  _SYSTEM_PROMPT = (
      "You are a precise GUI agent. Follow the user's instructions and "
      "emit exactly the format the user prompt requests."
  )

  def __init__(
      self,
      base_url: str = "http://127.0.0.1:4246/v1",
      model_name: str = "ui-tars-1.5-7b",
      api_key: str = "EMPTY",
      max_new_tokens: int = 512,
      temperature: float = 0.0,
      timeout: float = 180.0,
  ):
    try:
      from openai import OpenAI  # pylint: disable=import-outside-toplevel
    except ImportError as e:
      raise ImportError(
          "openai package required for UITarsVLLMWrapper. "
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
          f"[UI-TARS-vLLM] connected {base_url} | served={served}",
          flush=True,
      )
      if model_name not in served:
        print(
            f"[UI-TARS-vLLM] WARNING: '{model_name}' not in {served}. "
            f"Override --served-model-name on the server or pass a "
            f"matching value to the wrapper.",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
      raise RuntimeError(
          f"vLLM server unreachable at {base_url}: {type(e).__name__}: {e}\n"
          f"Start it first: scripts/start_ui_tars_vllm.sh"
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
      raise ImportError("Pillow required for UITarsVLLMWrapper.") from e

    self._call_no += 1
    call_no = self._call_no
    t0 = time.time()
    print(
        f"[UI-TARS-vLLM #{call_no}] predict_mm start  (images={len(images)})",
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
          f"[UI-TARS-vLLM #{call_no}] done in {elapsed:.1f}s  "
          f"in={in_tokens}  out={out_tokens}  | {preview}",
          flush=True,
      )
      return output_text, None, (output_text or "ok")

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[UI-TARS-vLLM #{call_no}] FAILED after {time.time()-t0:.1f}s: "
          f"{type(e).__name__}: {e}",
          flush=True,
      )
      logging.exception("[UI-TARS-vLLM] predict_mm failed")
      return ERROR_CALLING_LLM, None, None
