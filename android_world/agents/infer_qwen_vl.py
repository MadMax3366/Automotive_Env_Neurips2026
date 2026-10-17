"""Local Qwen2.5-VL wrapper conforming to infer.MultimodalLlmWrapper.

Imports torch / transformers lazily inside __init__, so non-GPU hosts can
still import this module without heavy deps.

Model is loaded once per instance — reused across predict_mm calls.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import numpy as np

from android_world.agents import infer

ERROR_CALLING_LLM = infer.ERROR_CALLING_LLM


class QwenVLWrapper(infer.LlmWrapper, infer.MultimodalLlmWrapper):
  """Local Qwen2.5-VL (HF transformers) wrapper for M3A / T3A."""

  # System prompt forces the strict M3A format. 3B-class models drift to
  # markdown / Chinese / extra preamble without explicit constraints, breaking
  # m3a_utils.parse_reason_action_output (`Reason:(.*)Action:` regex).
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
      model_path: str = "./models/Qwen2.5-VL-3B-Instruct",
      # 512 covers M3A action output (~50-200 tok) and summary output
      # (~50-150 tok) with headroom. 1024 was hurting tail latency when the
      # model over-ran format without hitting <|im_end|>.
      max_new_tokens: int = 512,
      min_pixels: int = 256 * 28 * 28,
      # Qwen2.5-VL officially recommended cap. 1280*28*28 ~ 1.0M px: AAOS AVD
      # screenshots (1408x792 ~ 1.1M px) get only a ~0.95x downscale so M3A's
      # bbox index labels stay readable, while avoiding the massive vision-
      # token expansion (and correspondingly slow processor._encode_plus) that
      # larger caps (e.g. 4096*28*28) produced.
      max_pixels: int = 1280 * 28 * 28,
      temperature: float = 0.0,
  ):
    if not os.path.isdir(model_path):
      raise FileNotFoundError(
          f"Model path not found: {model_path}. "
          f"Download first with: python3 scripts/download_qwen_vl.py"
      )

    # Heavy imports are local so the module itself is cheap to import.
    try:
      import torch
      from transformers import (  # pylint: disable=import-outside-toplevel
          AutoProcessor,
          Qwen2_5_VLForConditionalGeneration,
      )
    except ImportError as e:
      raise ImportError(
          "transformers + torch required for QwenVLWrapper. "
          "Install via requirements-gpu.txt."
      ) from e

    self._torch = torch
    self.model_path = model_path
    self.max_new_tokens = max_new_tokens
    self.temperature = temperature
    # Per-call counter + aggregate wall-time so operators can see live
    # progress under run_task.py which silences INFO-level logging.
    self._call_no = 0

    # Attention backend selection, in priority order:
    # 1. flash_attention_2 (fastest; needs `flash_attn` pkg + matching glibc/CUDA)
    # 2. sdpa                (torch's built-in memory-efficient attention; ~1.5-2x
    #                         over eager on bf16; available in torch >= 2.0,
    #                         always usable on CUDA; NO flash_attn dep)
    # 3. eager               (last-resort slow fallback)
    #
    # We probe flash_attn by *actually importing* its CUDA extension, because
    # the top-level `import flash_attn` can succeed while the underlying
    # `flash_attn_2_cuda.so` fails to load (e.g. glibc too old on the host).
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    try:
      from flash_attn import flash_attn_interface  # noqa: F401
      load_kwargs["attn_implementation"] = "flash_attention_2"
      # run_task.py pins logging to FATAL, so logging.info is swallowed.
      # Emit the backend selection to stdout so operators can verify it.
      print("[QwenVL] flash_attention_2 enabled", flush=True)
    except (ImportError, OSError) as e:
      load_kwargs["attn_implementation"] = "sdpa"
      print(
          f"[QwenVL] flash_attn unavailable ({type(e).__name__}) — falling back to SDPA",
          flush=True,
      )

    try:
      self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
          model_path, **load_kwargs
      )
    except torch.cuda.OutOfMemoryError as e:
      raise RuntimeError(
          f"OOM loading {model_path}. Consider a smaller model or free GPU memory."
      ) from e
    except Exception as e:
      # Retry once downgrading attention backend if that was the failure reason.
      if load_kwargs.get("attn_implementation") in ("flash_attention_2", "sdpa"):
        print(
            f"[QwenVL] load with {load_kwargs.get('attn_implementation')} "
            f"failed ({e}) — retrying with eager",
            flush=True,
        )
        load_kwargs.pop("attn_implementation", None)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        )
      else:
        raise

    self.model.eval()

    self.processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    logging.info(
        "[QwenVL] loaded %s on %s  max_new_tokens=%d  pixels=[%d..%d]",
        model_path,
        getattr(self.model, "device", "?"),
        max_new_tokens,
        min_pixels,
        max_pixels,
    )

  # ------------------------------------------------------------------ interface

  def predict(
      self, text_prompt: str
  ) -> tuple[str, Optional[bool], Any]:
    """Text-only call — no images."""
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Multimodal call. Returns (output_text, is_safe=None, raw=generated_ids)."""
    try:
      from PIL import Image
    except ImportError as e:  # should never happen; PIL is a transformers dep
      raise ImportError("Pillow required for QwenVLWrapper.") from e

    torch = self._torch

    self._call_no += 1
    call_no = self._call_no
    t0 = time.time()
    print(
        f"[QwenVL #{call_no}] predict_mm start  (images={len(images)})",
        flush=True,
    )

    try:
      pil_images = [Image.fromarray(img) for img in images]

      # Build Qwen chat messages. If no images, just a plain user text.
      content: list[dict] = []
      for pil in pil_images:
        content.append({"type": "image", "image": pil})
      content.append({"type": "text", "text": text_prompt})
      messages = [
          {"role": "system", "content": self._SYSTEM_PROMPT},
          {"role": "user", "content": content},
      ]

      chat_text = self.processor.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True
      )

      inputs = self.processor(
          text=[chat_text],
          images=pil_images if pil_images else None,
          videos=None,
          padding=True,
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

      # Trim away the prompt tokens.
      generated_trimmed = [
          out[len(inp):]
          for inp, out in zip(inputs.input_ids, generated_ids)
      ]
      output_text = self.processor.batch_decode(
          generated_trimmed,
          skip_special_tokens=True,
          clean_up_tokenization_spaces=False,
      )[0]

      elapsed = time.time() - t0
      in_tokens = int(inputs.input_ids.shape[1])
      out_tokens = int(len(generated_trimmed[0]))
      preview = " ".join((output_text or "").split())[:140]
      print(
          f"[QwenVL #{call_no}] done in {elapsed:.1f}s  "
          f"in={in_tokens}  out={out_tokens}  | {preview}",
          flush=True,
      )

      # NOTE: do not call torch.cuda.empty_cache() on the happy path.
      # It issues cudaDeviceSynchronize + cudaFree, burning ~100-300 ms
      # per call and forcing every subsequent alloc onto the slow
      # cudaMalloc path (cold cache). PyTorch's caching allocator reuses
      # blocks across steps for free; releasing them here destroys that
      # reuse. The OOM except below keeps a targeted empty_cache().
      # Return truthy non-tensor as raw so M3A's `if not raw_response:`
      # check does not hit `Boolean value of Tensor ambiguous`.
      return output_text, None, (output_text or "ok")

    except torch.cuda.OutOfMemoryError as e:
      print(
          f"[QwenVL #{call_no}] CUDA OOM after {time.time()-t0:.1f}s: {e}",
          flush=True,
      )
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      return ERROR_CALLING_LLM, None, None
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[QwenVL #{call_no}] FAILED after {time.time()-t0:.1f}s: "
          f"{type(e).__name__}: {e}",
          flush=True,
      )
      logging.exception("[QwenVL] predict_mm failed")
      return ERROR_CALLING_LLM, None, None
