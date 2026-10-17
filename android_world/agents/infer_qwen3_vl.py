"""Local Qwen3-VL wrapper conforming to infer.MultimodalLlmWrapper.

Mirrors infer_qwen_vl.QwenVLWrapper but loads the Qwen3-VL model class
(`Qwen3VLForConditionalGeneration`) instead of the 2.5-VL one. Chat-template
shape is the same (apply_chat_template handles the version difference).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import numpy as np

from android_world.agents import infer

ERROR_CALLING_LLM = infer.ERROR_CALLING_LLM


class Qwen3VLWrapper(infer.LlmWrapper, infer.MultimodalLlmWrapper):
  """Local Qwen3-VL (HF transformers) wrapper for M3A / T3A."""

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
      model_path: str = "./models/Qwen3-VL-8B-Instruct",
      max_new_tokens: int = 512,
      min_pixels: int = 256 * 28 * 28,
      max_pixels: int = 1280 * 28 * 28,
      temperature: float = 0.0,
  ):
    if not os.path.isdir(model_path):
      raise FileNotFoundError(
          f"Model path not found: {model_path}. "
          f"Download first with: python3 scripts/download_qwen3_vl.py"
      )

    try:
      import torch
      from transformers import AutoProcessor  # pylint: disable=import-outside-toplevel
      try:
        from transformers import (  # pylint: disable=import-outside-toplevel
            Qwen3VLForConditionalGeneration,
        )
      except ImportError as e:
        raise ImportError(
            "Qwen3VLForConditionalGeneration not found. Qwen3-VL requires "
            "transformers >= 4.57. Upgrade via "
            "`pip install -U git+https://github.com/huggingface/transformers`."
        ) from e
    except ImportError as e:
      raise ImportError(
          "transformers + torch required for Qwen3VLWrapper. "
          "Install via requirements-gpu.txt."
      ) from e

    self._torch = torch
    self.model_path = model_path
    self.max_new_tokens = max_new_tokens
    self.temperature = temperature
    self._call_no = 0

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    try:
      from flash_attn import flash_attn_interface  # noqa: F401
      load_kwargs["attn_implementation"] = "flash_attention_2"
      print("[Qwen3VL] flash_attention_2 enabled", flush=True)
    except (ImportError, OSError) as e:
      load_kwargs["attn_implementation"] = "sdpa"
      print(
          f"[Qwen3VL] flash_attn unavailable ({type(e).__name__}) — falling back to SDPA",
          flush=True,
      )

    try:
      self.model = Qwen3VLForConditionalGeneration.from_pretrained(
          model_path, **load_kwargs
      )
    except torch.cuda.OutOfMemoryError as e:
      raise RuntimeError(
          f"OOM loading {model_path}. Consider a smaller model or free GPU memory."
      ) from e
    except Exception as e:
      if load_kwargs.get("attn_implementation") in ("flash_attention_2", "sdpa"):
        print(
            f"[Qwen3VL] load with {load_kwargs.get('attn_implementation')} "
            f"failed ({e}) — retrying with eager",
            flush=True,
        )
        load_kwargs.pop("attn_implementation", None)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
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
        "[Qwen3VL] loaded %s on %s  max_new_tokens=%d  pixels=[%d..%d]",
        model_path,
        getattr(self.model, "device", "?"),
        max_new_tokens,
        min_pixels,
        max_pixels,
    )

  def predict(
      self, text_prompt: str
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    try:
      from PIL import Image
    except ImportError as e:
      raise ImportError("Pillow required for Qwen3VLWrapper.") from e

    torch = self._torch

    self._call_no += 1
    call_no = self._call_no
    t0 = time.time()
    print(
        f"[Qwen3VL #{call_no}] predict_mm start  (images={len(images)})",
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
          f"[Qwen3VL #{call_no}] done in {elapsed:.1f}s  "
          f"in={in_tokens}  out={out_tokens}  | {preview}",
          flush=True,
      )
      return output_text, None, (output_text or "ok")

    except torch.cuda.OutOfMemoryError as e:
      print(
          f"[Qwen3VL #{call_no}] CUDA OOM after {time.time()-t0:.1f}s: {e}",
          flush=True,
      )
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      return ERROR_CALLING_LLM, None, None
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[Qwen3VL #{call_no}] FAILED after {time.time()-t0:.1f}s: "
          f"{type(e).__name__}: {e}",
          flush=True,
      )
      logging.exception("[Qwen3VL] predict_mm failed")
      return ERROR_CALLING_LLM, None, None
