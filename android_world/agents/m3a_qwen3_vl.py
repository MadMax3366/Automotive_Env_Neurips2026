"""Local Qwen3-VL agent — reuses the M3A loop via Qwen3VLWrapper."""
from __future__ import annotations

from android_world.agents import infer_qwen3_vl
from android_world.agents import m3a
from android_world.env import interface


def build_m3a_qwen3_vl(
    env: interface.AsyncEnv,
    model_path: str = "./models/Qwen3-VL-8B-Instruct",
    max_new_tokens: int = 1024,
    name: str = "M3A-Qwen3VL",
) -> m3a.M3A:
  """Factory: M3A loop + local Qwen3-VL wrapper."""
  wrapper = infer_qwen3_vl.Qwen3VLWrapper(
      model_path=model_path,
      max_new_tokens=max_new_tokens,
  )
  return m3a.M3A(env, wrapper, name=name)
