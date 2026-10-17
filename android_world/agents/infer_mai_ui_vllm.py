"""Local MAI-UI wrapper that talks to a vLLM OpenAI-compatible server.

Same `predict_mm` contract as `infer_mai_ui.MAIUIWrapper`, but instead of
loading weights in-process it sends multimodal Chat Completions over
HTTP to a vLLM server (started by `scripts/start_mai_ui_vllm.sh`).
This is the official MAI-UI deployment recipe (vllm==0.11.0).

Why HTTP instead of in-process:
  - 4 emulators can share a single set of weights instead of paying
    16GB / GPU per emulator.
  - vLLM's continuous batching transparently coalesces concurrent
    `predict_mm` calls from multiple agent processes.
  - Server can stay warm across run_task.py invocations -- no 30-60s
    model load on every run.

Server is expected to expose at `base_url` (default
http://localhost:8000/v1) the same model name as `model_name`
(default MAI-UI-8B). This wrapper performs a one-shot reachability
check at __init__ and fails fast with a clear error if the server is
not up.
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


class MAIUIVLLMWrapper(infer.LlmWrapper, infer.MultimodalLlmWrapper):
  """vLLM-backed MAI-UI wrapper for M3A / T3A."""

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
      base_url: str = "http://localhost:8000/v1",
      model_name: str = "MAI-UI-8B",
      api_key: str = "EMPTY",
      max_new_tokens: int = 2048,
      temperature: float = 0.0,
      timeout: float = 120.0,
  ):
    try:
      from openai import OpenAI  # pylint: disable=import-outside-toplevel
    except ImportError as e:
      raise ImportError(
          "openai package required for MAIUIVLLMWrapper. "
          "Install via `pip install openai`."
      ) from e

    self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    self.base_url = base_url
    self.model_name = model_name
    self.max_new_tokens = max_new_tokens
    self.temperature = temperature
    self._call_no = 0

    # Fail fast if the server isn't reachable, with a useful hint.
    try:
      models = self._client.models.list()
      served = [m.id for m in (models.data or [])]
      print(
          f"[MAI-UI-vLLM] connected {base_url} | served={served}",
          flush=True,
      )
      if model_name not in served:
        print(
            f"[MAI-UI-vLLM] WARNING: '{model_name}' not in {served}. "
            f"Pass --model_endpoint or fix --served-model-name on the server.",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
      raise RuntimeError(
          f"vLLM server unreachable at {base_url}: {type(e).__name__}: {e}\n"
          f"Start it first:  bash scripts/start_mai_ui_vllm.sh"
      ) from e

  # ------------------------------------------------------------------ interface

  def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Send multimodal Chat Completion to vLLM. Returns (text, None, raw)."""
    try:
      from PIL import Image  # pylint: disable=import-outside-toplevel
    except ImportError as e:
      raise ImportError("Pillow required for MAIUIVLLMWrapper.") from e

    self._call_no += 1
    call_no = self._call_no
    t0 = time.time()
    print(
        f"[MAI-UI-vLLM #{call_no}] predict_mm start  (images={len(images)})",
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
          # MAI-UI's official runtime config: top_k=-1, top_p=1.0.
          # Passed via extra_body since OpenAI SDK doesn't expose top_k.
          extra_body={"top_k": -1, "top_p": 1.0},
      )

      output_text = response.choices[0].message.content or ""
      usage = response.usage
      in_tokens = int(getattr(usage, "prompt_tokens", 0)) if usage else 0
      out_tokens = int(getattr(usage, "completion_tokens", 0)) if usage else 0

      elapsed = time.time() - t0
      preview = " ".join(output_text.split())[:140]
      print(
          f"[MAI-UI-vLLM #{call_no}] done in {elapsed:.1f}s  "
          f"in={in_tokens}  out={out_tokens}  | {preview}",
          flush=True,
      )
      return output_text, None, (output_text or "ok")

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[MAI-UI-vLLM #{call_no}] FAILED after {time.time()-t0:.1f}s: "
          f"{type(e).__name__}: {e}",
          flush=True,
      )
      logging.exception("[MAI-UI-vLLM] predict_mm failed")
      return ERROR_CALLING_LLM, None, None
