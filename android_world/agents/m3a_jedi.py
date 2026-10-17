"""M3A loop tuned for Jedi-7B-1080p's `<tool_call>` JSON output.

Jedi was trained on the Anthropic Computer-Use tool-call schema:

    <tool_call>
    {"name": "computer", "arguments": {"action": "left_click",
                                        "coordinate": [598, 743]}}
    </tool_call>

Not pyautogui (OpenCUA), not M3A's "Reason:/Action: <JSON>", not native
key=value calls (UI-TARS). The English-description failure mode we
saw was Jedi falling back to free text when forced into M3A's prompt;
on its native prompt it emits the structured tool_call above.

This agent loop mirrors m3a_opencua.M3AOpenCUA but:
  * system + user prompts request the tool_call format
  * action parser extracts the inner JSON, maps Anthropic action names
    to JSONAction
  * smart_resize coord remap and stale-state warning are reused
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Optional

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCREENSHOT_DUMP_DIR = os.environ.get(
    "M3A_JEDI_SCREENSHOT_DIR",
    str(_PROJECT_ROOT / "screenshots" / "jedi"),
)


_JEDI_PROMPT_TEMPLATE = """## User Goal
{goal}

You are a GUI agent operating an **Android Automotive (AAOS) car infotainment screen** ({img_w}x{img_h} pixels). Layout typically:
  - left/right navigation rail with vehicle controls,
  - HVAC strip across the bottom (fan, A/C, temp setpoints, seat-heat, defrost, recirc, AUTO),
  - main content area in the middle.

For HVAC / climate / fan / temperature / seat / defrost goals, use the bottom HVAC strip — do NOT navigate into Media, Maps, or other apps unless the goal asks.

## Output Format
Emit ONE tool call inside `<tool_call>...</tool_call>` tags. No prose outside the tags. Schema:

<tool_call>
{{"name": "computer", "arguments": {{"action": "<action>", "coordinate": [x, y]}}}}
</tool_call>

Available actions (use coordinates in the {img_w}x{img_h} pixel space of THIS image):
  left_click           — click at (x, y)
  double_click         — double-click at (x, y)
  right_click          — right-click / long-press at (x, y)
  mouse_move           — hover; rarely needed on touch UI
  left_click_drag      — drag from current position to (x, y)
  type                 — type text; arguments include "text": "..."
  key                  — press a key; arguments include "text": "Home" | "Escape" | "Return"
  scroll               — arguments include "scroll_direction" ("up"/"down"/"left"/"right") and optional "scroll_amount"
Sentinels (no coordinate needed):
  done                 — task complete
  fail                 — task infeasible

## Repetition Guard
If your previous action did NOT change the screen, do NOT click the same coordinates. Try a different element or fall back to a `key`/`scroll`.

## Action History
{history}

## Reminder
{goal}

Now produce the next tool call."""


# ----------------------------------------------------------- output parser
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.+?)\s*</tool_call>", re.DOTALL)
_FALLBACK_JSON_RE = re.compile(
    r"\{[^{}]*\"action\"\s*:\s*\"[^\"]+\"[^{}]*\}", re.DOTALL
)


def _extract_tool_call_json(raw: str) -> Optional[dict]:
  """Pull the JSON payload out of a Jedi `<tool_call>` block. Falls
  back to a bare-JSON regex if the model omitted the tags (it
  sometimes does)."""
  if not raw:
    return None
  candidates = []
  m = _TOOL_CALL_RE.search(raw)
  if m:
    candidates.append(m.group(1))
  for jm in _FALLBACK_JSON_RE.finditer(raw):
    candidates.append(jm.group(0))
  for blob in candidates:
    try:
      return json.loads(blob)
    except (json.JSONDecodeError, ValueError):
      continue
  return None


def _parse_jedi_action(
    raw_output: str,
    screen_w: int, screen_h: int,
    resized_w: int, resized_h: int,
) -> Optional[json_action.JSONAction]:
  """Translate Jedi's <tool_call> JSON to a JSONAction.

  Coordinates emitted are in the model's smart-resized image space;
  rescale back to native AVD pixels.
  """
  payload = _extract_tool_call_json(raw_output)
  if not isinstance(payload, dict):
    return None

  args = payload.get("arguments")
  if not isinstance(args, dict):
    args = payload  # some variants put fields at top level
  action = (args.get("action") or "").strip().lower()
  if not action:
    return None

  def scale_xy(coord) -> Optional[tuple[int, int]]:
    if not (isinstance(coord, (list, tuple)) and len(coord) >= 2):
      return None
    try:
      x, y = int(coord[0]), int(coord[1])
    except (TypeError, ValueError):
      return None
    sx = int(round(x * screen_w / max(resized_w, 1)))
    sy = int(round(y * screen_h / max(resized_h, 1)))
    sx = max(0, min(screen_w - 1, sx))
    sy = max(0, min(screen_h - 1, sy))
    return sx, sy

  # Sentinels
  if action in ("done", "finished", "finish", "complete"):
    return json_action.JSONAction(action_type="status", goal_status="complete")
  if action in ("fail", "infeasible", "abort"):
    return json_action.JSONAction(action_type="status", goal_status="infeasible")

  if action in ("left_click", "click", "tap"):
    pt = scale_xy(args.get("coordinate"))
    if not pt:
      return None
    return json_action.JSONAction(action_type="click", x=pt[0], y=pt[1])

  if action == "double_click":
    pt = scale_xy(args.get("coordinate"))
    if not pt:
      return None
    return json_action.JSONAction(action_type="double_tap", x=pt[0], y=pt[1])

  if action in ("right_click", "long_press"):
    pt = scale_xy(args.get("coordinate"))
    if not pt:
      return None
    return json_action.JSONAction(action_type="long_press", x=pt[0], y=pt[1])

  if action in ("mouse_move", "hover"):
    pt = scale_xy(args.get("coordinate"))
    if not pt:
      return None
    return json_action.JSONAction(action_type="click", x=pt[0], y=pt[1])

  if action == "left_click_drag":
    # Best-effort: degrade to scroll in the dominant direction
    pt = scale_xy(args.get("coordinate"))
    if not pt:
      return None
    sx, sy = pt
    cx, cy = screen_w // 2, screen_h // 2
    dx, dy = sx - cx, sy - cy
    if abs(dx) >= abs(dy):
      return json_action.JSONAction(
          action_type="scroll", direction="right" if dx > 0 else "left"
      )
    return json_action.JSONAction(
        action_type="scroll", direction="down" if dy > 0 else "up"
    )

  if action == "type":
    text = args.get("text") or args.get("content") or ""
    if not isinstance(text, str):
      return None
    return json_action.JSONAction(action_type="input_text", text=text)

  if action == "key":
    key = (args.get("text") or args.get("key") or "").strip().lower()
    if key in ("home",):
      return json_action.JSONAction(action_type="navigate_home")
    if key in ("escape", "esc", "back"):
      return json_action.JSONAction(action_type="navigate_back")
    if key in ("return", "enter"):
      return json_action.JSONAction(action_type="keyboard_enter")
    return None

  if action == "scroll":
    direction = (args.get("scroll_direction") or args.get("direction") or "").lower()
    if direction in ("up", "down", "left", "right"):
      return json_action.JSONAction(action_type="scroll", direction=direction)
    return None

  return None


# Smart-resize verbatim from Qwen2.5-VL image_processing
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
class M3AJedi(base_agent.EnvironmentInteractingAgent):
  """M3A loop adapted for Jedi-7B-1080p's <tool_call> output."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm,
      name: str = "M3A-Jedi",
      *,
      max_history_steps: int = 8,
      wait_after_action_seconds: float = 2.0,
  ):
    super().__init__(env, name)
    self.llm = llm
    self._min_pixels = getattr(llm, "min_pixels", 256 * 28 * 28)
    self._max_pixels = getattr(llm, "max_pixels", 1280 * 28 * 28)
    self._max_history_steps = max_history_steps
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
      print(f"[m3a_jedi] saved {out}", flush=True)
    except Exception as e:  # noqa: BLE001
      print(f"[m3a_jedi] screenshot dump FAILED: "
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
          f"You issued click({last_act.x},{last_act.y}) {repeat_count} times "
          "and the screen has NOT changed. STOP clicking that pixel — output "
          "a different coordinate, a `key` (Home/Escape), or a `scroll`."
      )
    if screen_unchanged and last_act is not None:
      return (
          "\n\n## FEEDBACK\n"
          f"Your last click at ({last_act.x},{last_act.y}) did NOT change "
          "the screen. Pick a different element."
      )
    if repeat_count >= 2 and last_act is not None:
      return (
          "\n\n## FEEDBACK\n"
          f"Stop repeating ({last_act.x},{last_act.y}); pick a new element."
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

    prompt = _JEDI_PROMPT_TEMPLATE.format(
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
    step_data["action_text"] = action_output[:300]

    json_act = _parse_jedi_action(
        action_output, screen_w, screen_h, resized_w, resized_h,
    )
    if json_act is None:
      step_data["summary"] = (
          f"Could not parse Jedi tool_call: {action_output[:120]!r}"
      )
      logging.info("[m3a_jedi] %s", step_data["summary"])
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)
    step_data["action_output_json"] = json_act

    try:
      self.env.execute_action(json_act)
    except Exception as e:  # noqa: BLE001
      step_data["summary"] = f"Failed to execute action: {e}"
      logging.exception("[m3a_jedi] action exec failed")
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    time.sleep(self.wait_after_action_seconds)
    step_data["summary"] = f"Action: {action_output[:160]}"
    self.history.append(step_data)

    is_done = (json_act.action_type == "status")
    return base_agent.AgentInteractionResult(is_done, step_data)
