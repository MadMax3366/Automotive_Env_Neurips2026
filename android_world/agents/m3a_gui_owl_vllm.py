"""M3A loop driven by GUI-Owl-7B served via vLLM HTTP server.

GUI-Owl is a Qwen2.5-VL-7B-Instruct fine-tune; the M3A loop and its
"Reason:/Action:" prompt format work as-is. We just swap the in-process
QwenVLWrapper for the HTTP-only GUIOwlVLLMWrapper.
"""
from __future__ import annotations

from android_world.agents import infer_gui_owl_vllm
from android_world.agents import m3a
from android_world.env import interface


def build_m3a_gui_owl_vllm(
    env: interface.AsyncEnv,
    base_url: str = "http://127.0.0.1:4244/v1",
    model_name: str = "gui-owl-7b",
    max_new_tokens: int = 1024,
    name: str = "M3A-GUI-Owl-vLLM",
) -> m3a.M3A:
  """Factory: M3A loop + remote GUI-Owl via vLLM."""
  wrapper = infer_gui_owl_vllm.GUIOwlVLLMWrapper(
      base_url=base_url,
      model_name=model_name,
      max_new_tokens=max_new_tokens,
  )
  return m3a.M3A(env, wrapper, name=name)
