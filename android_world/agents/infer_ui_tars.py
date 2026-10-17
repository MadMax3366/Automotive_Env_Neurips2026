"""Local ByteDance-Seed/UI-TARS-1.5-7B wrapper.

UI-TARS-1.5 is a 8.3B image-text-to-text model purpose-built for GUI
agents. Architecture is `qwen2_5_vl` (compatible with our existing
Qwen wrapper code), but we follow the official ByteDance / HF model
card pattern of `AutoModelForImageTextToText` + the unified
`apply_chat_template(return_dict=True, tokenize=True)` path so the
model's bundled chat template is used as-is.

Same M3A integration shape as infer_qwen_vl / infer_mai_ui:
  - implements `predict_mm` returning (text, None, raw)
  - system prompt forces M3A's `Reason:` / `Action:` format
  - per-call `[UI-TARS #N]` progress on stdout
  - flash_attention_2 -> SDPA -> eager fallback chain
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import numpy as np

from android_world.agents import infer

ERROR_CALLING_LLM = infer.ERROR_CALLING_LLM


class UITarsWrapper(infer.LlmWrapper, infer.MultimodalLlmWrapper):
  """Local UI-TARS-1.5-7B (HF transformers) wrapper for M3A / T3A."""

  # UI-TARS-1.5 was trained with `Thought: ... Action: ...` headers
  # (see ByteDance's MOBILE_USE_DOUBAO / COMPUTER_USE_DOUBAO templates
  # at github.com/bytedance/UI-TARS/blob/main/codes/ui_tars/prompt.py).
  # Asking for `Reason:` is out-of-distribution and the model collapses
  # to the GROUNDING_DOUBAO path that emits a bare `Action: ...` with no
  # reasoning, which then breaks M3A's strict
  # `re.search(r'Reason:(.*)Action:')` parser.
  #
  # We honour the model's native format (Thought + Action) and rewrite
  # `Thought:` -> `Reason:` post-hoc so M3A's parser sees what it
  # expects without us touching m3a_utils.
  _SYSTEM_PROMPT = (
      "You are a precise Android Automotive GUI agent. "
      "When answering an action selection prompt you MUST output EXACTLY two lines, "
      "in English, with no markdown, no code fences, no preamble, no extra text:\n"
      "Thought: <one short sentence explaining what you are about to do and why>\n"
      "Action: <one JSON object on a single line>\n"
      "Always emit the literal English words `Thought:` and `Action:` with this "
      "capitalization and ordering. The JSON must be valid and contain "
      "`action_type`. Do not wrap the JSON in backticks. Do not output any "
      "other label like `Observation:` or `Reasoning:`."
  )

  def __init__(
      self,
      model_path: str = "./models/UI-TARS-1.5-7B",
      max_new_tokens: int = 512,
      min_pixels: int = 256 * 28 * 28,
      max_pixels: int = 1280 * 28 * 28,
      temperature: float = 0.0,
      # When True (default, paired with M3A's index-based prompt), rewrite
      # `Thought:` -> `Reason:` post-hoc so M3A's parser is satisfied.
      # Set False when paired with the native UI-TARS agent
      # (M3AUITarsNative) which parses `Thought:` / `Action:` directly.
      rewrite_thought_to_reason: bool = True,
      # When False, also skip the placeholder Reason injection for bare
      # `Action:` outputs (the native agent treats those as parse errors).
      inject_placeholder_reason: bool = True,
  ):
    if not os.path.isdir(model_path):
      raise FileNotFoundError(
          f"Model path not found: {model_path}. "
          f"Download first with: python3 scripts/download_ui_tars.py"
      )

    try:
      import torch
      from transformers import (  # pylint: disable=import-outside-toplevel
          AutoProcessor,
          AutoModelForImageTextToText,
      )
    except ImportError as e:
      raise ImportError(
          "transformers + torch required for UITarsWrapper. "
          "Install via requirements-gpu.txt."
      ) from e

    self._torch = torch
    self.model_path = model_path
    self.max_new_tokens = max_new_tokens
    self.temperature = temperature
    self._rewrite_thought_to_reason = rewrite_thought_to_reason
    self._inject_placeholder_reason = inject_placeholder_reason
    self._call_no = 0

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    try:
      from flash_attn import flash_attn_interface  # noqa: F401
      load_kwargs["attn_implementation"] = "flash_attention_2"
      print("[UI-TARS] flash_attention_2 enabled", flush=True)
    except (ImportError, OSError) as e:
      load_kwargs["attn_implementation"] = "sdpa"
      print(
          f"[UI-TARS] flash_attn unavailable ({type(e).__name__}) — falling back to SDPA",
          flush=True,
      )

    try:
      self.model = AutoModelForImageTextToText.from_pretrained(
          model_path, **load_kwargs
      )
    except torch.cuda.OutOfMemoryError as e:
      raise RuntimeError(
          f"OOM loading {model_path}. Consider freeing GPU memory."
      ) from e
    except Exception as e:
      if load_kwargs.get("attn_implementation") in ("flash_attention_2", "sdpa"):
        print(
            f"[UI-TARS] load with {load_kwargs.get('attn_implementation')} "
            f"failed ({e}) — retrying with eager",
            flush=True,
        )
        load_kwargs.pop("attn_implementation", None)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, **load_kwargs
        )
      else:
        raise

    self.model.eval()

    try:
      self.processor = AutoProcessor.from_pretrained(
          model_path, min_pixels=min_pixels, max_pixels=max_pixels,
      )
    except TypeError:
      self.processor = AutoProcessor.from_pretrained(model_path)

    logging.info(
        "[UI-TARS] loaded %s on %s  max_new_tokens=%d  pixels=[%d..%d]",
        model_path,
        getattr(self.model, "device", "?"),
        max_new_tokens,
        min_pixels,
        max_pixels,
    )

  # ------------------------------------------------------------------ interface

  def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    try:
      from PIL import Image
    except ImportError as e:
      raise ImportError("Pillow required for UITarsWrapper.") from e

    torch = self._torch
    self._call_no += 1
    call_no = self._call_no
    t0 = time.time()
    print(
        f"[UI-TARS #{call_no}] predict_mm start  (images={len(images)})",
        flush=True,
    )

    try:
      pil_images = [Image.fromarray(img) for img in images]
      content: list[dict] = []
      for pil in pil_images:
        content.append({"type": "image", "image": pil})
      content.append({"type": "text", "text": text_prompt})
      messages = [
          {"role": "system", "content": self._SYSTEM_PROMPT},
          {"role": "user", "content": content},
      ]

      inputs = self.processor.apply_chat_template(
          messages,
          add_generation_prompt=True,
          tokenize=True,
          return_dict=True,
          return_tensors="pt",
      ).to(self.model.device)

      gen_kwargs = dict(
          max_new_tokens=self.max_new_tokens,
          do_sample=self.temperature > 0,
      )
      if self.temperature > 0:
        gen_kwargs["temperature"] = self.temperature

      with torch.no_grad():
        generated_ids = self.model.generate(**inputs, **gen_kwargs)

      input_len = int(inputs["input_ids"].shape[-1])
      trimmed = generated_ids[:, input_len:]
      output_text = self.processor.batch_decode(
          trimmed,
          skip_special_tokens=True,
          clean_up_tokenization_spaces=False,
      )[0]

      # When paired with the M3A index-based agent: bridge UI-TARS's
      # native format to M3A's expected `Reason:` / `Action:` schema.
      # When paired with M3AUITarsNative: skip both rewrites so the
      # agent can parse the native Thought / Action format directly.
      if self._rewrite_thought_to_reason:
        if "Thought:" in output_text and "Reason:" not in output_text:
          output_text = output_text.replace("Thought:", "Reason:", 1)
        if (self._inject_placeholder_reason
            and "Reason:" not in output_text and "Action:" in output_text):
          output_text = "Reason: UI-TARS direct action.\n" + output_text.lstrip()

      elapsed = time.time() - t0
      out_tokens = int(trimmed.shape[-1])
      preview = " ".join((output_text or "").split())[:140]
      print(
          f"[UI-TARS #{call_no}] done in {elapsed:.1f}s  "
          f"in={input_len}  out={out_tokens}  | {preview}",
          flush=True,
      )
      return output_text, None, (output_text or "ok")

    except torch.cuda.OutOfMemoryError as e:
      print(
          f"[UI-TARS #{call_no}] CUDA OOM after {time.time()-t0:.1f}s: {e}",
          flush=True,
      )
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      return ERROR_CALLING_LLM, None, None
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[UI-TARS #{call_no}] FAILED after {time.time()-t0:.1f}s: "
          f"{type(e).__name__}: {e}",
          flush=True,
      )
      logging.exception("[UI-TARS] predict_mm failed")
      return ERROR_CALLING_LLM, None, None
