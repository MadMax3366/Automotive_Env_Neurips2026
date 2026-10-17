"""Local OpenCUA-7B wrapper for M3A.

OpenCUA models (xlangai/OpenCUA-{7B,32B,72B}) are computer-use foundation
models built on Qwen2.5-VL-7B-Instruct, with two architecture changes:

  - M-RoPE replaced with 1D RoPE
  - Tokenizer + chat template aligned with Kimi-VL

Both are exposed via custom code in the HF repo, so loading requires
`trust_remote_code=True`. The output format is pyautogui code, e.g.::

    pyautogui.click(x=1443, y=343)
    pyautogui.scroll(0, 3, x=712, y=400)

Coordinates are in the SMART-RESIZED image space (Qwen2.5-VL processor's
post-resize pixels). The agent loop in m3a_opencua.py is responsible for
mapping back to native screen pixels via smart_resize.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import numpy as np

from android_world.agents import infer

ERROR_CALLING_LLM = infer.ERROR_CALLING_LLM


def _patch_opencua_tie_weights_signature(model_path: str) -> None:
  """Make OpenCUA's bundled `modeling_opencua.py` compatible with
  modern transformers.

  Symptom (transformers >= 4.50ish):
      TypeError: OpenCUAForConditionalGeneration.tie_weights() got an
      unexpected keyword argument 'recompute_mapping'

  Root cause: OpenCUA's custom modeling code defines
      def tie_weights(self):
  but newer transformers calls
      self.tie_weights(recompute_mapping=False)
  during PreTrainedModel.init_weights(). The override doesn't accept
  any kwarg, so the call blows up before the model finishes loading.

  Fix: rewrite the signature in-place to accept **kwargs (which the
  override then ignores, matching the previous no-arg behaviour). Two
  files get patched if present:

    1. <model_path>/modeling_opencua.py — the canonical source. New
       processes loading this model_path will pick up the patched copy
       on the next `trust_remote_code` cache rebuild.
    2. <HF cache>/modules/transformers_modules/<repo>/modeling_opencua.py
       — the hashed copy transformers uses at import time. Already-
       cached stale copies from prior failed loads need patching too,
       otherwise the import path stays broken.

  Idempotent: if the new signature is already in place, no-op. Silent
  on other signatures (different upstream version) so we don't corrupt
  unrelated code.
  """
  targets: list[str] = []
  src_path = os.path.join(model_path, "modeling_opencua.py")
  if os.path.isfile(src_path):
    targets.append(src_path)

  # HF caches custom modeling_*.py under $HF_HOME/modules/transformers_modules/.
  # Cover both the env-overridden HF_HOME and the default ~/.cache/huggingface
  # so this works whether the user set HF_HOME explicitly or not.
  cache_roots = []
  hf_home = os.environ.get("HF_HOME")
  if hf_home:
    cache_roots.append(os.path.join(hf_home, "modules", "transformers_modules"))
  cache_roots.append(
      os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
  )
  for root_dir in cache_roots:
    if not os.path.isdir(root_dir):
      continue
    for root, _, files in os.walk(root_dir):
      base = os.path.basename(root).lower()
      if "opencua" not in base:
        continue
      for f in files:
        if f == "modeling_opencua.py":
          targets.append(os.path.join(root, f))

  patched = []
  for fp in dict.fromkeys(targets):  # de-dup, preserve order
    try:
      with open(fp, "r", encoding="utf-8") as fh:
        text = fh.read()
      if "def tie_weights(self, **kwargs):" in text:
        continue  # already patched
      if "def tie_weights(self):" not in text:
        continue  # different signature — leave alone
      new_text = text.replace(
          "def tie_weights(self):",
          "def tie_weights(self, **kwargs):",
          1,
      )
      with open(fp, "w", encoding="utf-8") as fh:
        fh.write(new_text)
      patched.append(fp)
    except Exception as e:  # noqa: BLE001
      print(
          f"[OpenCUA] tie_weights patch skipped for {fp}: "
          f"{type(e).__name__}: {e}",
          flush=True,
      )

  if patched:
    for fp in patched:
      print(
          f"[OpenCUA] patched tie_weights signature in {fp}",
          flush=True,
      )


class OpenCUAWrapper(infer.LlmWrapper, infer.MultimodalLlmWrapper):
  """Local OpenCUA wrapper. Same lazy-import / SDPA-fallback pattern
  as the other local-VL wrappers."""

  # OpenCUA's own paper / vLLM example uses this exact system prompt for
  # GUI grounding. Keep it verbatim — the model was trained against it.
  _SYSTEM_PROMPT = (
      "You are a GUI agent. You are given a task and a screenshot of "
      "the screen. You need to perform a series of pyautogui actions "
      "to complete the task."
  )

  def __init__(
      self,
      model_path: str = "./models/OpenCUA-7B",
      max_new_tokens: int = 512,
      min_pixels: int = 256 * 28 * 28,
      max_pixels: int = 1280 * 28 * 28,
      temperature: float = 0.0,
  ):
    if not os.path.isdir(model_path):
      raise FileNotFoundError(
          f"Model path not found: {model_path}. "
          f"Download first with: python3 scripts/download_opencua.py"
      )

    # Apply the upstream-compat patch BEFORE transformers gets a chance
    # to import the trust-remote-code module. If the cache already has
    # a stale (un-patched) copy from a previous failed load, this also
    # rewrites it in place so the module re-import picks up the fix.
    _patch_opencua_tie_weights_signature(model_path)

    try:
      import torch
      from transformers import AutoModel, AutoProcessor, AutoTokenizer
    except ImportError as e:
      raise ImportError(
          "transformers + torch required for OpenCUAWrapper."
      ) from e

    self._torch = torch
    self.model_path = model_path
    self.max_new_tokens = max_new_tokens
    self.temperature = temperature
    self._call_no = 0

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    try:
      from flash_attn import flash_attn_interface  # noqa: F401
      load_kwargs["attn_implementation"] = "flash_attention_2"
      print("[OpenCUA] flash_attention_2 enabled", flush=True)
    except (ImportError, OSError) as e:
      load_kwargs["attn_implementation"] = "sdpa"
      print(
          f"[OpenCUA] flash_attn unavailable ({type(e).__name__}) — "
          f"falling back to SDPA",
          flush=True,
      )

    try:
      self.model = AutoModel.from_pretrained(model_path, **load_kwargs)
    except torch.cuda.OutOfMemoryError as e:
      raise RuntimeError(
          f"OOM loading {model_path}. Free GPU memory or pick a "
          f"smaller variant."
      ) from e
    except Exception as e:
      if load_kwargs.get("attn_implementation") in ("flash_attention_2", "sdpa"):
        print(
            f"[OpenCUA] load with {load_kwargs.get('attn_implementation')} "
            f"failed ({e}) — retrying with eager",
            flush=True,
        )
        load_kwargs.pop("attn_implementation", None)
        self.model = AutoModel.from_pretrained(model_path, **load_kwargs)
      else:
        raise

    self.model.eval()

    # Processor (image preprocessing + smart_resize) and tokenizer
    # (Kimi-VL-aligned chat template) are both required.
    self.processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    # Cache pixel caps for the agent's smart_resize math.
    self.min_pixels = min_pixels
    self.max_pixels = max_pixels

    logging.info(
        "[OpenCUA] loaded %s on %s  max_new_tokens=%d  pixels=[%d..%d]",
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
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    try:
      from PIL import Image
    except ImportError as e:
      raise ImportError("Pillow required for OpenCUAWrapper.") from e

    torch = self._torch
    self._call_no += 1
    call_no = self._call_no
    t0 = time.time()
    print(
        f"[OpenCUA #{call_no}] predict_mm start  (images={len(images)})",
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

      # OpenCUA's processor returns a few extra keys that the bundled
      # custom forward()/generate() don't accept. Newer transformers
      # rejects any unknown kwarg with:
      #   ValueError: The following `model_kwargs` are not used by the
      #   model: ['mm_token_type_ids'] ...
      # Strip the known-bad ones rather than monkey-patching the model.
      # This list grows as we discover more — kept centralized here so a
      # future processor change is easy to follow.
      _OPENCUA_DROP_KEYS = (
          "mm_token_type_ids",
      )
      for k in _OPENCUA_DROP_KEYS:
        if hasattr(inputs, "pop"):
          inputs.pop(k, None)
        elif k in inputs:
          del inputs[k]

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
          f"[OpenCUA #{call_no}] done in {elapsed:.1f}s  "
          f"in={in_tokens}  out={out_tokens}  | {preview}",
          flush=True,
      )
      return output_text, None, (output_text or "ok")

    except torch.cuda.OutOfMemoryError as e:
      print(f"[OpenCUA #{call_no}] CUDA OOM after {time.time()-t0:.1f}s: {e}",
            flush=True)
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      return ERROR_CALLING_LLM, None, None
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[OpenCUA #{call_no}] FAILED after {time.time()-t0:.1f}s: "
          f"{type(e).__name__}: {e}",
          flush=True,
      )
      logging.exception("[OpenCUA] predict_mm failed")
      return ERROR_CALLING_LLM, None, None
