"""M3A OpenCUA agent loop driven by a remote vLLM server.

Reuses the M3AOpenCUA loop from m3a_opencua.py (pyautogui parsing,
smart-resize coord scaling, stale-state warning) and only swaps the
in-process HF wrapper for OpenCUAVLLMWrapper.
"""
from __future__ import annotations

from android_world.agents import infer_opencua_vllm
from android_world.agents import m3a_opencua
from android_world.env import interface


def build_m3a_opencua_vllm(
    env: interface.AsyncEnv,
    base_url: str = "http://127.0.0.1:4243/v1",
    model_name: str = "opencua-7b",
    max_new_tokens: int = 512,
    name: str = "M3A-OpenCUA-vLLM",
    *,
    min_pixels: int = 3136,
    max_pixels: int = 2007040,
) -> m3a_opencua.M3AOpenCUA:
  """Factory: M3AOpenCUA loop + remote vLLM wrapper."""
  wrapper = infer_opencua_vllm.OpenCUAVLLMWrapper(
      base_url=base_url,
      model_name=model_name,
      max_new_tokens=max_new_tokens,
      min_pixels=min_pixels,
      max_pixels=max_pixels,
  )
  return m3a_opencua.M3AOpenCUA(env, wrapper, name=name)
