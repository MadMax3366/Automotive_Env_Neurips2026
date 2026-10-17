"""Local Qwen2.5-VL agent — reuses the M3A loop via QwenVLWrapper.

M3A is wrapper-agnostic (accepts any MultimodalLlmWrapper), so we don't need
to duplicate its 600-line step loop. Just wire up the local wrapper.
"""
from __future__ import annotations

from android_world.agents import infer_qwen_vl
from android_world.agents import m3a
from android_world.env import interface


def build_m3a_qwen_vl(
    env: interface.AsyncEnv,
    model_path: str = "./models/Qwen2.5-VL-3B-Instruct",
    max_new_tokens: int = 1024,
    name: str = "M3A-QwenVL",
) -> m3a.M3A:
  """Factory: M3A loop + local Qwen2.5-VL wrapper."""
  wrapper = infer_qwen_vl.QwenVLWrapper(
      model_path=model_path,
      max_new_tokens=max_new_tokens,
  )
  return m3a.M3A(env, wrapper, name=name)
