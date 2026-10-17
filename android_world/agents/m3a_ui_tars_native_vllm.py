"""M3A native-coordinate loop driven by UI-TARS-1.5-7B served via vLLM.

Reuses the existing M3AUITarsNative agent class (native click(point=...)
parser, smart-resize coord remap, stale-state warning) and only swaps
the in-process UITarsWrapper for the HTTP UITarsVLLMWrapper.
"""
from __future__ import annotations

from android_world.agents import infer_ui_tars_vllm
from android_world.agents import m3a_ui_tars_native
from android_world.env import interface


def build_m3a_ui_tars_native_vllm(
    env: interface.AsyncEnv,
    base_url: str = "http://127.0.0.1:4246/v1",
    model_name: str = "ui-tars-1.5-7b",
    max_new_tokens: int = 512,
    name: str = "M3A-UI-TARS-Native-vLLM",
    *,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> m3a_ui_tars_native.M3AUITarsNative:
  """Factory: M3AUITarsNative loop + remote UI-TARS via vLLM."""
  wrapper = infer_ui_tars_vllm.UITarsVLLMWrapper(
      base_url=base_url,
      model_name=model_name,
      max_new_tokens=max_new_tokens,
  )
  return m3a_ui_tars_native.M3AUITarsNative(
      env, wrapper, name=name,
      min_pixels=min_pixels, max_pixels=max_pixels,
  )
