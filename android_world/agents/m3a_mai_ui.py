"""Local MAI-UI agent — reuses the M3A loop via MAIUIWrapper.

M3A is wrapper-agnostic (accepts any MultimodalLlmWrapper), so we don't
need to duplicate its 600-line step loop. Just wire up the local
Tongyi-MAI/MAI-UI wrapper.
"""
from __future__ import annotations

from android_world.agents import infer_mai_ui
from android_world.agents import m3a
from android_world.env import interface


def build_m3a_mai_ui(
    env: interface.AsyncEnv,
    model_path: str = "./models/MAI-UI-2B",
    max_new_tokens: int = 512,
    name: str = "M3A-MAI-UI",
) -> m3a.M3A:
  """Factory: M3A loop + local MAI-UI wrapper."""
  wrapper = infer_mai_ui.MAIUIWrapper(
      model_path=model_path,
      max_new_tokens=max_new_tokens,
  )
  return m3a.M3A(env, wrapper, name=name)
