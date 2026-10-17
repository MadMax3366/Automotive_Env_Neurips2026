"""Local MiMo-VL agent — reuses the M3A loop via MiMoVLWrapper.

M3A is wrapper-agnostic (accepts any MultimodalLlmWrapper), so we don't
need to duplicate its 600-line step loop. Just wire up the local
XiaomiMiMo/MiMo-VL wrapper.
"""
from __future__ import annotations

from android_world.agents import infer_mimo_vl
from android_world.agents import m3a
from android_world.env import interface


def build_m3a_mimo_vl(
    env: interface.AsyncEnv,
    model_path: str = "./models/MiMo-VL-7B-RL",
    max_new_tokens: int = 512,
    name: str = "M3A-MiMoVL",
) -> m3a.M3A:
  """Factory: M3A loop + local MiMo-VL wrapper."""
  wrapper = infer_mimo_vl.MiMoVLWrapper(
      model_path=model_path,
      max_new_tokens=max_new_tokens,
  )
  return m3a.M3A(env, wrapper, name=name)
