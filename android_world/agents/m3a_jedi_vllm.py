"""M3A loop driven by Jedi-7B-1080p served via vLLM.

Uses the dedicated M3AJedi class with `<tool_call>` JSON parser since
Jedi was trained on the Anthropic Computer-Use tool schema, not
pyautogui (OpenCUA) and not M3A's "Reason:/Action: <JSON>". Empirical
evidence: Jedi emits

    <tool_call>
    {"name": "computer", "arguments": {"action": "left_click",
                                        "coordinate": [598, 743]}}
    </tool_call>

verbatim when given its native prompt.
"""
from __future__ import annotations

from android_world.agents import infer_jedi_vllm
from android_world.agents import m3a_jedi
from android_world.env import interface


def build_m3a_jedi_vllm(
    env: interface.AsyncEnv,
    base_url: str = "http://127.0.0.1:4245/v1",
    model_name: str = "jedi-7b-1080p",
    max_new_tokens: int = 512,
    name: str = "M3A-Jedi-vLLM",
) -> m3a_jedi.M3AJedi:
  """Factory: M3AJedi loop (<tool_call> parser + coord remap) +
  remote Jedi via vLLM."""
  wrapper = infer_jedi_vllm.JediVLLMWrapper(
      base_url=base_url,
      model_name=model_name,
      max_new_tokens=max_new_tokens,
  )
  return m3a_jedi.M3AJedi(env, wrapper, name=name)
