"""M3A variant tuned for OpenCUA-7B's pyautogui-style action format.

OpenCUA emits pyautogui code blocks per step, e.g.::

    ```python
    pyautogui.click(x=1443, y=343)
    ```

Coordinates are in the SMART-RESIZED image space (Qwen2.5-VL processor).
We rescale back to AAOS native screen pixels before executing.

Design follows m3a_ui_tars_native:
  * Strip M3A bbox marks; OpenCUA wants the raw screenshot
  * Custom system prompt + per-step instruction (the model card's
    "GUI agent + pyautogui actions" recipe)
  * Coordinate parsing via regex for click / scroll / typewrite /
    press / hotkey / moveTo + a `finished` / `done` sentinel
  * Stale-state warning identical to ui_tars_native (md5 hash of
    screenshot + repeat-coordinate detector) since OpenCUA is also
    greedy-decoded and prone to dead-loop on a missed click
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Optional

from android_world.agents import base_agent
from android_world.agents import infer_opencua
from android_world.env import interface
from android_world.env import json_action

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCREENSHOT_DUMP_DIR = os.environ.get(
    "M3A_OPENCUA_SCREENSHOT_DIR",
    str(_PROJECT_ROOT / "screenshots" / "opencua"),
)


# ----------------------------------------------------------------- prompt
# OpenCUA was trained for desktop OS (Ubuntu / Windows / macOS). The
# AAOS infotainment surface is unfamiliar; we lean hard on the goal +
# AAOS layout description, like we did for UI-TARS native.
_OPENCUA_PROMPT_TEMPLATE = """## User Goal
{goal}

You are a GUI agent operating an **Android Automotive (AAOS) car infotainment screen**, presented as a {img_w}x{img_h} pixel image. Layout typically:
  - left/right navigation rail with vehicle controls,
  - HVAC strip across the bottom (fan, A/C, temp setpoints, seat-heat, defrost, recirc, AUTO),
  - main content area in the middle.

For HVAC / climate / fan / temperature / seat / defrost goals, use the bottom HVAC strip — do NOT navigate into Media, Maps, or other apps unless the goal asks.

## Output Format
Emit ONE pyautogui call inside a fenced ```python ... ``` block. No prose, no extra text.

Available actions (use coordinates in the {img_w}x{img_h} pixel space of THIS image):
  pyautogui.click(x=N, y=N)
  pyautogui.doubleClick(x=N, y=N)
  pyautogui.rightClick(x=N, y=N)
  pyautogui.moveTo(x=N, y=N)               # rare; only if hover required
  pyautogui.scroll(amount, x=N, y=N)       # +amount = up, -amount = down
  pyautogui.hscroll(amount, x=N, y=N)      # +amount = right, -amount = left
  pyautogui.dragTo(x=N, y=N, duration=0.5)
  pyautogui.typewrite('text', interval=0)
  pyautogui.press('enter' | 'backspace' | 'tab' | 'home' | 'esc')
  pyautogui.hotkey('ctrl', 'l')
Sentinels (no pyautogui prefix):
  done()                                    # task complete
  fail()                                    # task infeasible

## Repetition Guard
If your previous action did NOT change the screen, do NOT click the same coordinates. Try a clearly different element or a new region.

## Action History
{history}

## Reminder
{goal}

Now produce the next pyautogui action."""


# ----------------------------------------------------------- output parser
_PY_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
_PY_LINE_RE = re.compile(
    r"(pyautogui\.\w+\([^)]*\)|done\s*\(\s*\)|fail\s*\(\s*\))"
)


def _extract_pyautogui_call(raw: str) -> Optional[str]:
  """Find the single pyautogui call (or done()/fail()) in the model
  output. Prefer code-fenced text but fall back to a bare line match
  since OpenCUA occasionally drops the fence."""
  if not raw:
    return None
  candidates: list[str] = []
  for m in _PY_BLOCK_RE.finditer(raw):
    candidates.append(m.group(1))
  candidates.append(raw)  # last-resort: treat the whole raw output as code
  for blob in candidates:
    m2 = _PY_LINE_RE.search(blob)
    if m2:
      return m2.group(1).strip()
  return None


_INT_KW_RE = lambda kw: re.compile(rf"\b{kw}\s*=\s*(-?\d+)")
_X_RE = _INT_KW_RE("x")
_Y_RE = _INT_KW_RE("y")
_AMOUNT_POSITIONAL_RE = re.compile(r"\(\s*(-?\d+)")  # first positional int


def _parse_xy(call: str) -> Optional[tuple[int, int]]:
  mx = _X_RE.search(call)
  my = _Y_RE.search(call)
  if mx and my:
    return int(mx.group(1)), int(my.group(1))
  return None


def _parse_pyautogui_action(
    call: str,
    screen_w: int, screen_h: int,
    resized_w: int, resized_h: int,
) -> Optional[json_action.JSONAction]:
  """Translate one pyautogui.* call to a JSONAction. Coordinates emitted
  by OpenCUA are in the smart-resized space; rescale back to native."""
  if not call:
    return None
  call = call.strip()

  def scale_xy(x: int, y: int) -> tuple[int, int]:
    sx = int(round(x * screen_w / max(resized_w, 1)))
    sy = int(round(y * screen_h / max(resized_h, 1)))
    sx = max(0, min(screen_w - 1, sx))
    sy = max(0, min(screen_h - 1, sy))
    return sx, sy

  # Sentinels first
  if re.match(r"^done\s*\(", call):
    return json_action.JSONAction(action_type="status", goal_status="complete")
  if re.match(r"^fail\s*\(", call):
    return json_action.JSONAction(action_type="status", goal_status="infeasible")

  # Identify pyautogui method
  method_m = re.match(r"pyautogui\.(\w+)\s*\(", call)
  if not method_m:
    return None
  method = method_m.group(1).lower()

  if method in ("click", "leftclick", "doubleclick", "tripleclick"):
    pt = _parse_xy(call)
    if not pt:
      # Some OpenCUA outputs use bare positional pyautogui.click(1443, 343)
      m = re.match(r"pyautogui\.\w+\s*\(\s*(-?\d+)\s*,\s*(-?\d+)", call)
      if m:
        pt = (int(m.group(1)), int(m.group(2)))
    if not pt:
      return None
    sx, sy = scale_xy(*pt)
    return json_action.JSONAction(action_type="click", x=sx, y=sy)

  if method in ("rightclick",):
    pt = _parse_xy(call)
    if not pt:
      return None
    sx, sy = scale_xy(*pt)
    # No native right-click in m3a/AAOS; fall back to long_press.
    return json_action.JSONAction(action_type="long_press", x=sx, y=sy)

  if method == "moveto":
    # No useful "hover" action — treat as click as a best-effort.
    pt = _parse_xy(call)
    if not pt:
      return None
    sx, sy = scale_xy(*pt)
    return json_action.JSONAction(action_type="click", x=sx, y=sy)

  if method in ("dragto",):
    pt = _parse_xy(call)
    if not pt:
      return None
    # Treat drag as a scroll in the dominant direction (degrades
    # gracefully when the drag is meant as a swipe).
    sx, sy = scale_xy(*pt)
    cx, cy = screen_w // 2, screen_h // 2
    dx, dy = sx - cx, sy - cy
    if abs(dx) >= abs(dy):
      return json_action.JSONAction(
          action_type="scroll", direction="right" if dx > 0 else "left"
      )
    return json_action.JSONAction(
        action_type="scroll", direction="down" if dy > 0 else "up"
    )

  if method == "scroll":
    m = _AMOUNT_POSITIONAL_RE.search(call)
    if not m:
      return None
    amt = int(m.group(1))
    return json_action.JSONAction(
        action_type="scroll", direction="up" if amt > 0 else "down"
    )

  if method == "hscroll":
    m = _AMOUNT_POSITIONAL_RE.search(call)
    if not m:
      return None
    amt = int(m.group(1))
    return json_action.JSONAction(
        action_type="scroll", direction="right" if amt > 0 else "left"
    )

  if method in ("typewrite", "write"):
    tm = re.search(r"['\"](.*?)['\"]", call, re.DOTALL)
    if not tm:
      return None
    return json_action.JSONAction(action_type="input_text", text=tm.group(1))

  if method == "press":
    tm = re.search(r"['\"](\w+)['\"]", call)
    if not tm:
      return None
    key = tm.group(1).lower()
    if key in ("home",):
      return json_action.JSONAction(action_type="navigate_home")
    if key in ("escape", "esc", "back"):
      return json_action.JSONAction(action_type="navigate_back")
    if key in ("enter", "return"):
      return json_action.JSONAction(action_type="keyboard_enter")
    # Unmapped key — best-effort no-op.
    return None

  if method == "hotkey":
    # Combined modifiers — best-effort: if it includes 'home' or 'esc'
    # treat as nav, else no-op.
    if "'home'" in call.lower():
      return json_action.JSONAction(action_type="navigate_home")
    if "'esc'" in call.lower() or "'escape'" in call.lower():
      return json_action.JSONAction(action_type="navigate_back")
    return None

  return None


# Smart-resize (verbatim from Qwen2.5-VL image_processing_qwen2_vl.py)
def _smart_resize_qwen(
    height: int, width: int, factor: int = 28,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> tuple[int, int]:
  if height < factor or width < factor:
    raise ValueError(f"smart_resize: image too small {height}x{width}")
  if max(height, width) / min(height, width) > 200:
    raise ValueError("smart_resize: extreme aspect ratio")
  h_bar = round(height / factor) * factor
  w_bar = round(width / factor) * factor
  if h_bar * w_bar > max_pixels:
    beta = math.sqrt((height * width) / max_pixels)
    h_bar = math.floor(height / beta / factor) * factor
    w_bar = math.floor(width / beta / factor) * factor
  elif h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (height * width))
    h_bar = math.ceil(height * beta / factor) * factor
    w_bar = math.ceil(width * beta / factor) * factor
  return h_bar, w_bar


# ----------------------------------------------------------- agent class
class M3AOpenCUA(base_agent.EnvironmentInteractingAgent):
  """M3A loop adapted for OpenCUA's pyautogui output."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer_opencua.OpenCUAWrapper,
      name: str = "M3A-OpenCUA",
      *,
      max_history_steps: int = 8,
      wait_after_action_seconds: float = 2.0,
  ):
    super().__init__(env, name)
    self.llm = llm
    self._min_pixels = llm.min_pixels
    self._max_pixels = llm.max_pixels
    self._max_history_steps = max_history_steps
    # Sleep gap between sending an action and reading the next a11y
    # tree / screenshot. Without this attribute set, step() crashes on
    # `time.sleep(self.wait_after_action_seconds)` -> AttributeError ->
    # entire task fails after the first successful click. Matches the
    # m3a / m3a_ui_tars_native default of 2.0s.
    self.wait_after_action_seconds = wait_after_action_seconds
    self.history: list[dict] = []
    self._current_task_id = ""

  def reset(self, go_home: bool = False) -> None:
    super().reset(go_home)
    self.history = []

  def _dump_screenshot(self, arr, step_idx: int) -> None:
    if not _SCREENSHOT_DUMP_DIR:
      return
    try:
      from PIL import Image
      task_safe = re.sub(r"[^\w-]", "_", self._current_task_id or "task")[:60]
      Path(_SCREENSHOT_DUMP_DIR).mkdir(parents=True, exist_ok=True)
      out = Path(_SCREENSHOT_DUMP_DIR) / f"{task_safe}_step{step_idx:02d}.png"
      Image.fromarray(arr).save(out)
      print(f"[m3a_opencua] saved {out}", flush=True)
    except Exception as e:  # noqa: BLE001
      print(f"[m3a_opencua] screenshot dump FAILED: "
            f"{type(e).__name__}: {e}", flush=True)

  @staticmethod
  def _img_hash(arr) -> str:
    try:
      return hashlib.md5(arr.tobytes()).hexdigest()
    except Exception:  # noqa: BLE001
      return ""

  def _stale_state_warning(self, current_hash: str) -> str:
    if not self.history:
      return ""
    last = self.history[-1]
    last_act = last.get("action_output_json")
    last_hash = last.get("_screen_hash") or ""
    repeat_count = 0
    if last_act is not None:
      for prev in reversed(self.history):
        pact = prev.get("action_output_json")
        if (pact is not None
            and pact.action_type == last_act.action_type
            and pact.x == last_act.x and pact.y == last_act.y):
          repeat_count += 1
        else:
          break
    screen_unchanged = bool(current_hash) and current_hash == last_hash
    if repeat_count >= 2 and screen_unchanged:
      return (
          "\n\n## CRITICAL FEEDBACK\n"
          f"You have already issued the same click({last_act.x},{last_act.y}) "
          f"{repeat_count} times in a row and the screen has NOT changed. "
          "STOP clicking that pixel — output a different coordinate, a "
          "press('home'), or a press('back')."
      )
    if screen_unchanged and last_act is not None:
      return (
          "\n\n## FEEDBACK\n"
          f"Your last click at ({last_act.x},{last_act.y}) did NOT change "
          "the screen. Pick a different element this turn."
      )
    if repeat_count >= 2 and last_act is not None:
      return (
          "\n\n## FEEDBACK\n"
          f"Stop repeating the same coordinate ({last_act.x},{last_act.y}); "
          "pick a new element."
      )
    return ""

  def _build_history(self) -> str:
    if not self.history:
      return "(no actions taken yet)"
    rows = self.history[-self._max_history_steps:]
    lines = []
    for i, h in enumerate(rows, 1):
      atext = (h.get("action_text") or "").strip().replace("\n", " ")[:200]
      lines.append(f"Step {i}: {atext}")
    return "\n".join(lines)

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data: dict = {
        "raw_screenshot": None,
        "action_prompt": None,
        "action_output": None,
        "action_output_json": None,
        "action_raw_response": None,
        "action_reason": None,
        "thought": None,
        "action_text": None,
        "summary": None,
    }

    state = self.get_post_transition_state()
    screenshot = state.pixels.copy()
    screen_h, screen_w = screenshot.shape[:2]
    step_data["raw_screenshot"] = screenshot
    cur_hash = self._img_hash(screenshot)
    step_data["_screen_hash"] = cur_hash
    self._current_task_id = goal[:60]
    self._dump_screenshot(screenshot, len(self.history) + 1)
    stale_rider = self._stale_state_warning(cur_hash)

    resized_h, resized_w = _smart_resize_qwen(
        screen_h, screen_w,
        min_pixels=self._min_pixels, max_pixels=self._max_pixels,
    )

    prompt = _OPENCUA_PROMPT_TEMPLATE.format(
        img_w=resized_w, img_h=resized_h,
        history=self._build_history(),
        goal=goal,
    )
    if stale_rider:
      prompt += stale_rider
    step_data["action_prompt"] = prompt

    action_output, _is_safe, raw = self.llm.predict_mm(prompt, [screenshot])
    if not raw:
      raise RuntimeError("Error calling LLM in action selection phase.")
    if action_output is None:
      action_output = ""
    step_data["action_output"] = action_output
    step_data["action_raw_response"] = raw

    call = _extract_pyautogui_call(action_output)
    step_data["action_text"] = call or ""
    step_data["thought"] = ""  # OpenCUA in our prompt doesn't separate thought
    step_data["action_reason"] = ""

    if not call:
      step_data["summary"] = (
          f"Output had no pyautogui call: {action_output[:120]!r}"
      )
      logging.info("[m3a_opencua] %s", step_data["summary"])
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    json_act = _parse_pyautogui_action(
        call, screen_w, screen_h, resized_w, resized_h,
    )
    if json_act is None:
      step_data["summary"] = f"Could not parse pyautogui call: {call[:120]}"
      logging.info("[m3a_opencua] %s", step_data["summary"])
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)
    step_data["action_output_json"] = json_act

    try:
      self.env.execute_action(json_act)
    except Exception as e:  # noqa: BLE001
      step_data["summary"] = f"Failed to execute action: {e}"
      logging.exception("[m3a_opencua] action exec failed")
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    time.sleep(self.wait_after_action_seconds)
    step_data["summary"] = f"Action: {call[:160]}"
    self.history.append(step_data)

    is_done = (json_act.action_type == "status")
    return base_agent.AgentInteractionResult(is_done, step_data)


# ----------------------------------------------------------------- factory
def build_m3a_opencua(
    env: interface.AsyncEnv,
    model_path: str = "./models/OpenCUA-7B",
    max_new_tokens: int = 512,
    name: str = "M3A-OpenCUA",
) -> M3AOpenCUA:
  wrapper = infer_opencua.OpenCUAWrapper(
      model_path=model_path,
      max_new_tokens=max_new_tokens,
  )
  return M3AOpenCUA(env, wrapper, name=name)
