"""M3A variant tuned for UI-TARS-1.5's native action format.

UI-TARS-1.5 was trained on the MOBILE_USE_DOUBAO / COMPUTER_USE_DOUBAO
prompt templates, where the agent emits:

    Thought: <one short sentence>
    Action: click(point='<point>x y</point>')   # or long_press / scroll / type / open_app / press_home / press_back / wait / finished

i.e. **coordinate-based** actions, not the bbox-index actions M3A's
default prompt asks for. Forcing UI-TARS into the index-based schema
visibly degrades quality — the model spatially mis-identifies which
labeled bbox is "home button" / "fan up" etc., and with greedy
decoding it loops on the same wrong index for ~5 wasted steps per
task (observed empirically on AAOS HVAC v3).

This agent fixes that by:

  * stripping the bbox marks from the screenshot (UI-TARS gets the raw
    image — its training distribution),
  * replacing the M3A action-selection prompt with a UI-TARS native
    template (Thought + native action enumeration),
  * parsing the native action and converting it to a coordinate-based
    `JSONAction` (M3A's executor already supports `x` / `y` for
    `click` / `long_press` / `input_text`, and bare `direction` for
    `scroll`),
  * scaling the model's output coordinates from the processor's
    smart-resize space back to the screen's native pixel space, and
  * skipping the M3A summary LLM call entirely — UI-TARS's `Thought`
    is itself a per-step record we can replay as text history,
    halving wall-clock per step.

The wrapper is the same `UITarsWrapper` we already ship; this agent
just constructs it with `rewrite_thought_to_reason=False` so the
post-processing doesn't get in the way.
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
from android_world.agents import infer
from android_world.agents import infer_ui_tars
from android_world.env import interface
from android_world.env import json_action

# Where to dump every input screenshot the native agent feeds to
# UI-TARS, named `<task_id>_step<N>.png`. Default lives inside the
# project tree so operators can vscode-open it without scp; override
# with M3A_UI_TARS_SCREENSHOT_DIR. Set to empty string to disable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCREENSHOT_DUMP_DIR = os.environ.get(
    "M3A_UI_TARS_SCREENSHOT_DIR",
    str(_PROJECT_ROOT / "screenshots" / "uitars_native"),
)


# ----------------------------------------------------------------- prompt
# Tuned for Android Automotive (AAOS) where UI-TARS-1.5 (trained mostly
# on phones / desktops) tends to mis-identify the AAOS launcher rail
# as a "Local Media Player" or generic phone app drawer and dead-loop
# clicking it. The fixes vs. a vanilla MOBILE_USE_DOUBAO prompt:
#   - Goal moved to the very TOP and reiterated at the bottom.
#   - Explicit AAOS context: not a phone, has HVAC strip, layout differs.
#   - Negative instruction: don't open Media/Maps unless goal says so.
#   - Repetition guard: if last click didn't change UI, try elsewhere.
_NATIVE_PROMPT_TEMPLATE = """## User Goal (PRIMARY OBJECTIVE)
{goal}

You are a GUI agent operating an **Android Automotive (AAOS) car infotainment screen** — NOT a smartphone, NOT a tablet. The screen layout typically has:
  - a left or right navigation rail with vehicle controls (climate, settings, apps, etc.),
  - a persistent HVAC strip along the BOTTOM edge with a fan icon, A/C toggle, temperature setpoints, and seat-heater / defroster controls,
  - a main content area in the middle.

Climate, HVAC, fan, temperature, seat-heater, defroster, recirc, and AUTO goals are reached through that HVAC strip — DO NOT navigate to "Local Media Player", "Maps", or any other app unless the goal text explicitly asks for it.

## Output Format
You MUST emit EXACTLY two lines, in English, with no markdown, no code fences, and no extra labels:

Thought: <one short sentence: what you will tap and why, grounded in the user goal above>
Action: <one native action call from the action space below>

## Action Space
click(point='<point>x y</point>')                    # tap at pixel (x, y) of the input image
long_press(point='<point>x y</point>')               # long-press at pixel (x, y)
type(content='<text content>')                       # type text into the currently focused field
scroll(point='<point>x y</point>', direction='down') # direction: 'up' | 'down' | 'left' | 'right'
open_app(app_name='<app name>')                      # open an installed app by name
press_home()                                         # navigate to the home screen
press_back()                                         # navigate back
wait()                                               # wait for the screen to update
finished()                                           # mark the user goal as complete

Coordinates: emit absolute pixel coordinates of the input image you see. The image is approximately {img_w}x{img_h} pixels.

## Repetition Guard
If your previous action did NOT change the screen (the screenshot looks the same as before), DO NOT repeat the same coordinates. Try a clearly different element, a different region of the screen, or fall back to `press_back()` / `press_home()` to re-orient yourself.

## Action History
{history}

## Reminder of the User Goal
{goal}

Now produce the next Thought and Action, focused on completing the user goal above."""


# ----------------------------------------------------------- output parser
_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n\s*Action:|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.*)", re.DOTALL)
# Action call: name(arg=...)
_CALL_RE = re.compile(r"^\s*(\w+)\s*\((.*)\)\s*$", re.DOTALL)


def _extract_point(args: str) -> Optional[tuple[int, int]]:
  """Pull (x, y) out of UI-TARS native point argument.

  Tongyi's MOBILE_USE_DOUBAO docs show `<point>x y</point>`, but
  UI-TARS-1.5-7B in practice emits any of:
    point='<point>120 340</point>'
    point='<point>120, 340</point>'
    point='(120, 340)'
    point='(120,340)'
    point='120,340'
  Probe each in order, return the first hit.
  """
  for pattern in (
      r"<point>\s*(-?\d+)\s*[,\s]+\s*(-?\d+)\s*</point>",
      r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
      r"(-?\d+)\s*,\s*(-?\d+)",
  ):
    m = re.search(pattern, args)
    if m:
      return int(m.group(1)), int(m.group(2))
  return None


def _smart_resize_qwen(
    h: int, w: int,
    factor: int = 28,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> tuple[int, int]:
  """Reproduces the Qwen2.5-VL processor's smart_resize step so we can
  scale model-emitted coordinates back to screen pixels.

  Returns (resized_h, resized_w) — both multiples of `factor`.
  """
  if h * w <= 0:
    return h, w
  h_bar = max(factor, round(h / factor) * factor)
  w_bar = max(factor, round(w / factor) * factor)
  if h_bar * w_bar > max_pixels:
    beta = math.sqrt((h * w) / max_pixels)
    h_bar = max(factor, math.floor(h / beta / factor) * factor)
    w_bar = max(factor, math.floor(w / beta / factor) * factor)
  elif h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (h * w))
    h_bar = math.ceil(h * beta / factor) * factor
    w_bar = math.ceil(w * beta / factor) * factor
  return h_bar, w_bar


def _parse_native_action(
    action_text: str,
    screen_w: int, screen_h: int,
    resized_w: int, resized_h: int,
) -> Optional[json_action.JSONAction]:
  """Parse a single UI-TARS native action call to a JSONAction.

  Coordinate scaling: model emits pixel coords in the processor's
  resized image space (resized_w x resized_h). We map back to the
  AVD's native screen pixel space (screen_w x screen_h) so M3A's
  click/long_press path can tap the right spot.
  """
  m = _CALL_RE.match(action_text)
  if not m:
    return None
  name = m.group(1).lower().strip()
  args = m.group(2)

  def scale_xy(x: int, y: int) -> tuple[int, int]:
    sx = int(round(x * screen_w / max(resized_w, 1)))
    sy = int(round(y * screen_h / max(resized_h, 1)))
    sx = max(0, min(screen_w - 1, sx))
    sy = max(0, min(screen_h - 1, sy))
    return sx, sy

  if name in ("click", "left_single", "left_double", "right_single"):
    pt = _extract_point(args)
    if pt is None:
      return None
    sx, sy = scale_xy(pt[0], pt[1])
    return json_action.JSONAction(action_type="click", x=sx, y=sy)

  if name == "long_press":
    pt = _extract_point(args)
    if pt is None:
      return None
    sx, sy = scale_xy(pt[0], pt[1])
    return json_action.JSONAction(action_type="long_press", x=sx, y=sy)

  if name == "type":
    cm = re.search(r"content\s*=\s*['\"](.*?)['\"]", args, re.DOTALL)
    if not cm:
      return None
    return json_action.JSONAction(action_type="input_text", text=cm.group(1))

  if name == "scroll":
    dm = re.search(r"direction\s*=\s*['\"](\w+)['\"]", args)
    if not dm:
      return None
    direction = dm.group(1).lower()
    if direction not in ("up", "down", "left", "right"):
      return None
    return json_action.JSONAction(action_type="scroll", direction=direction)

  if name == "open_app":
    am = re.search(r"app_name\s*=\s*['\"](.*?)['\"]", args, re.DOTALL)
    if not am:
      return None
    return json_action.JSONAction(action_type="open_app", app_name=am.group(1))

  if name in ("press_home", "navigate_home"):
    return json_action.JSONAction(action_type="navigate_home")

  if name in ("press_back", "navigate_back"):
    return json_action.JSONAction(action_type="navigate_back")

  if name == "wait":
    return json_action.JSONAction(action_type="wait")

  if name in ("finished", "finish", "done"):
    return json_action.JSONAction(
        action_type="status", goal_status="complete"
    )

  if name == "drag":
    # Best-effort: map drag start/end to a scroll direction.
    pts: list[tuple[int, int]] = []
    for pattern in (
        r"<point>\s*(-?\d+)\s*[,\s]+\s*(-?\d+)\s*</point>",
        r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
    ):
      pts.extend((int(a), int(b)) for a, b in re.findall(pattern, args))
      if len(pts) >= 2:
        break
    if len(pts) >= 2:
      x1, y1 = pts[0]
      x2, y2 = pts[1]
      dx, dy = x2 - x1, y2 - y1
      if abs(dx) >= abs(dy):
        return json_action.JSONAction(
            action_type="scroll", direction="right" if dx > 0 else "left",
        )
      return json_action.JSONAction(
          action_type="scroll", direction="down" if dy > 0 else "up",
      )
    return None

  return None


# ------------------------------------------------------------------- agent
class M3AUITarsNative(base_agent.EnvironmentInteractingAgent):
  """M3A loop variant tuned for UI-TARS-1.5 native coordinate actions."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      name: str = "M3AUITarsNative",
      wait_after_action_seconds: float = 2.0,
      min_pixels: int = 256 * 28 * 28,
      max_pixels: int = 1280 * 28 * 28,
  ):
    super().__init__(env, name)
    self.llm = llm
    self.history: list[dict] = []
    self.additional_guidelines = None
    self.wait_after_action_seconds = wait_after_action_seconds
    self._min_pixels = min_pixels
    self._max_pixels = max_pixels
    self._current_task_id: str = "task"
    if _SCREENSHOT_DUMP_DIR:
      print(
          f"[m3a_ui_tars_native] screenshot dump enabled -> "
          f"{_SCREENSHOT_DUMP_DIR} (set M3A_UI_TARS_SCREENSHOT_DIR='' to disable)",
          flush=True,
      )

  def reset(self, go_home_on_reset: bool = False) -> None:
    super().reset(go_home_on_reset)
    try:
      self.env.hide_automation_ui()
    except Exception:  # noqa: BLE001
      pass
    self.history = []
    self._current_task_id: str = "task"  # set at step() time via goal text

  def _dump_screenshot(self, screenshot, step_no: int) -> None:
    """Save the input screenshot so we can inspect what UI-TARS saw."""
    if not _SCREENSHOT_DUMP_DIR:
      return
    try:
      from PIL import Image  # local import; PIL not always needed
      out_dir = Path(_SCREENSHOT_DUMP_DIR)
      out_dir.mkdir(parents=True, exist_ok=True)
      slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(self._current_task_id))[:80]
      path = out_dir / f"{slug}_step{step_no:02d}.png"
      Image.fromarray(screenshot).save(path)
      print(f"[m3a_ui_tars_native] saved {path}", flush=True)
    except Exception as e:  # noqa: BLE001 - dump is diagnostic only
      print(
          f"[m3a_ui_tars_native] screenshot dump FAILED: {type(e).__name__}: {e}",
          flush=True,
      )

  @staticmethod
  def _img_hash(arr) -> str:
    """Cheap content hash for change detection."""
    try:
      return hashlib.md5(arr.tobytes()).hexdigest()
    except Exception:  # noqa: BLE001
      return ""

  def _stale_state_warning(self, current_hash: str) -> str:
    """If the screen and/or the last action are visibly stuck, emit a
    one-shot prompt rider telling UI-TARS to break out.

    Three increasingly insistent levels:
      - last action repeated >=2 times AND screen unchanged: bail
        the position entirely.
      - screen unchanged only: the click landed on dead pixels.
      - last action repeated only: the click got somewhere but the
        model is still issuing the same one.
    """
    if not self.history:
      return ""
    last = self.history[-1]
    last_act = last.get("action_output_json")
    last_hash = last.get("_screen_hash") or ""

    # action repetition window: last action and the one before that
    repeat_count = 0
    if last_act is not None:
      for prev in reversed(self.history):
        pact = prev.get("action_output_json")
        if (pact is not None
            and pact.action_type == last_act.action_type
            and pact.x == last_act.x and pact.y == last_act.y
            and pact.index == last_act.index):
          repeat_count += 1
        else:
          break

    screen_unchanged = bool(current_hash) and current_hash == last_hash

    if repeat_count >= 2 and screen_unchanged:
      desc = (
          f"click({last_act.x}, {last_act.y})" if last_act else "<unknown>"
      )
      return (
          "\n\n## CRITICAL FEEDBACK\n"
          f"You have already issued `{desc}` {repeat_count} times in a row "
          "and the screen has NOT changed. That position is dead. "
          "STOP clicking it. Output a clearly different action this turn — "
          "either a different coordinate, a `press_back()`, or a "
          "`press_home()` to re-orient."
      )
    if screen_unchanged and last_act is not None:
      return (
          "\n\n## FEEDBACK\n"
          f"Your last action `click({last_act.x}, {last_act.y})` did NOT "
          "change the screen. Pick a different element this turn."
      )
    if repeat_count >= 2 and last_act is not None:
      return (
          "\n\n## FEEDBACK\n"
          f"You have repeated the same `click({last_act.x}, {last_act.y})` "
          f"{repeat_count} times. The screen IS responding to clicks elsewhere "
          "— do NOT keep emitting the same coordinate. Pick a new element."
      )
    return ""

  def _build_history(self) -> str:
    if not self.history:
      return "(no actions taken yet)"
    lines = []
    for i, h in enumerate(self.history, 1):
      thought = (h.get("thought") or "").strip().replace("\n", " ")[:200]
      atext = (h.get("action_text") or "").strip().replace("\n", " ")[:200]
      lines.append(f"Step {i} Thought: {thought}")
      lines.append(f"Step {i} Action: {atext}")
    return "\n".join(lines)

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data: dict = {
        "raw_screenshot": None,
        "action_prompt": None,
        "action_output": None,
        "action_output_json": None,
        "action_raw_response": None,
        # M3A-compatible fields so run_task.py's trace collector finds
        # them by name instead of falling back to
        # m3a_utils.parse_reason_action_output (which was producing
        # "No JSON match in click(point=...)" noise on every step).
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

    prompt = _NATIVE_PROMPT_TEMPLATE.format(
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
    step_data["action_output"] = action_output
    step_data["action_raw_response"] = raw

    thought_m = _THOUGHT_RE.search(action_output)
    action_m = _ACTION_RE.search(action_output)
    thought = thought_m.group(1).strip() if thought_m else ""
    action_text = action_m.group(1).strip() if action_m else ""
    # Action: line might span multiple lines if the model added a
    # trailing comment. Keep only the first non-empty line.
    if action_text:
      action_text = action_text.splitlines()[0].strip()
    step_data["thought"] = thought
    step_data["action_text"] = action_text
    # Mirror thought into action_reason so run_task.py's trace
    # collector skips the m3a_utils fallback parser.
    step_data["action_reason"] = thought

    if not action_text:
      step_data["summary"] = "Output missing Action: line; no action performed."
      logging.info("[m3a_ui_tars_native] %s", step_data["summary"])
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    json_act = _parse_native_action(
        action_text, screen_w, screen_h, resized_w, resized_h,
    )
    if json_act is None:
      step_data["summary"] = (
          f"Could not parse native UI-TARS action: {action_text[:120]}"
      )
      logging.info("[m3a_ui_tars_native] %s", step_data["summary"])
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)
    step_data["action_output_json"] = json_act

    try:
      self.env.execute_action(json_act)
    except Exception as e:  # noqa: BLE001
      step_data["summary"] = f"Failed to execute action: {e}"
      logging.exception("[m3a_ui_tars_native] action exec failed")
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    time.sleep(self.wait_after_action_seconds)

    step_data["summary"] = (
        f"Thought: {thought[:160]}  Action: {action_text[:160]}"
    )
    self.history.append(step_data)

    is_done = (json_act.action_type == "status")
    return base_agent.AgentInteractionResult(is_done, step_data)


# ----------------------------------------------------------------- factory
def build_m3a_ui_tars_native(
    env: interface.AsyncEnv,
    model_path: str = "./models/UI-TARS-1.5-7B",
    max_new_tokens: int = 512,
    name: str = "M3A-UI-TARS-Native",
) -> M3AUITarsNative:
  """Factory: M3A native loop + UI-TARS wrapper without Thought->Reason rewrite."""
  wrapper = infer_ui_tars.UITarsWrapper(
      model_path=model_path,
      max_new_tokens=max_new_tokens,
      rewrite_thought_to_reason=False,
      inject_placeholder_reason=False,
  )
  return M3AUITarsNative(env, wrapper, name=name)
