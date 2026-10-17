"""Local UI-TARS agent — reuses the M3A loop via UITarsWrapper.

M3A is wrapper-agnostic; we only swap the LLM backend.
"""
from __future__ import annotations

from android_world.agents import infer_ui_tars
from android_world.agents import m3a
from android_world.env import interface


def build_m3a_ui_tars(
    env: interface.AsyncEnv,
    model_path: str = "./models/UI-TARS-1.5-7B",
    max_new_tokens: int = 512,
    name: str = "M3A-UI-TARS",
) -> m3a.M3A:
  """Factory: M3A loop + local UI-TARS wrapper."""
  wrapper = infer_ui_tars.UITarsWrapper(
      model_path=model_path,
      max_new_tokens=max_new_tokens,
  )
  return m3a.M3A(env, wrapper, name=name)
