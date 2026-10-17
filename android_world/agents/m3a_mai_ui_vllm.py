"""M3A loop driven by MAI-UI served via vLLM HTTP server.

Wrapper-agnostic: M3A doesn't care that the LLM is remote. We just
swap in MAIUIVLLMWrapper.
"""
from __future__ import annotations

from android_world.agents import infer_mai_ui_vllm
from android_world.agents import m3a
from android_world.env import interface


def build_m3a_mai_ui_vllm(
    env: interface.AsyncEnv,
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "MAI-UI-8B",
    max_new_tokens: int = 2048,
    name: str = "M3A-MAI-UI-vLLM",
) -> m3a.M3A:
  """Factory: M3A loop + remote MAI-UI via vLLM."""
  wrapper = infer_mai_ui_vllm.MAIUIVLLMWrapper(
      base_url=base_url,
      model_name=model_name,
      max_new_tokens=max_new_tokens,
  )
  return m3a.M3A(env, wrapper, name=name)
