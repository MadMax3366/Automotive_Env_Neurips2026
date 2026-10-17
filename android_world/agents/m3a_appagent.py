"""M3A variant using AppAgent-style prompt structure.

Goals (per user spec):
  * Prompt drawn from AppAgent's `task_template` (introduces the
    numbered-bbox UI representation, function call notation for
    tap/text/long_press/swipe, optional <ui_document> doc-augmentation
    placeholder for future explore-phase docs).
  * Output format kept compatible with M3A's parser:
        Reason: <one short sentence>
        Action: {"action_type": "...", "index": N, ...}
    so we reuse `m3a_utils.parse_reason_action_output` and the rest of
    M3A's execution / validator / trace machinery.
  * Bbox numbering + screenshot annotation reused from
    `m3a_utils.add_ui_element_mark` (M3A's existing renderer); element
    list comes from android_env's a11y tree (NOT AppAgent's
    uiautomator XML), so element indices align with M3A.
  * No-doc mode is the default — explore phase is optional. When
    `--appagent_docs` is set later, docs get spliced into the
    `<ui_document>` block.
  * Full M3A action space exposed (click/long_press/scroll/input_text/
    navigate_back/navigate_home/keyboard_enter/status), beyond
    AppAgent's native (tap/text/long_press/swipe/FINISH) — required
    so v3 tasks like infeasibility detection work.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import json_action

# Where docs from explore phase live. Layout:
#   <APPAGENT_DOCS_DIR>/<elem_uid>.txt
# Each .txt is AppAgent-native JSON-style:
#   {"tap": "...", "long_press": "...", "text": "...",
#    "v_swipe": "...", "h_swipe": "..."}
# An env override lets users point at an alternate doc set without
# touching code.
_DEFAULT_DOCS_DIR = os.environ.get(
    "M3A_APPAGENT_DOCS_DIR", ""  # empty = no-doc mode
)


# ----------------------------------------------------------------- prompt
# Adapted from externel_agent/AppAgent/scripts/prompts.task_template.
# Replaced AppAgent's tap()/swipe() output instructions with M3A's
# JSON-action requirement, and extended action space to cover
# navigate_back / navigate_home / status / scroll(direction) so v3
# tasks (HVAC + safety/infeasible) are fully expressible.
_APPAGENT_PROMPT_TEMPLATE = """You are an agent that performs tasks on an Android Automotive (AAOS) infotainment screen. You will be given a screenshot of the screen with interactive UI elements labeled by numeric tags starting from 1. The numeric tag of each interactive element is located at the center of the element.

You can interact with the screen via the following action types. Always respond using EXACTLY two lines, in English, with no markdown or code fences:

    Reason: <one short sentence: what you will do and why, grounded in the user goal>
    Action: <one JSON object on a single line>

The JSON object must contain `action_type` and other fields as appropriate:

  - {{"action_type": "click", "index": <int>}}
      Tap on the UI element with the given numeric tag.
  - {{"action_type": "long_press", "index": <int>}}
      Long-press the UI element with the given numeric tag.
  - {{"action_type": "input_text", "text": "<string>", "index": <int>}}
      Type text into the input element with the given numeric tag.
      Provide `index` so we know which input field to focus.
  - {{"action_type": "scroll", "direction": "<up|down|left|right>"}}
      Scroll the screen content in the given direction.
  - {{"action_type": "navigate_home"}}
      Press the system Home button to return to the home screen.
  - {{"action_type": "navigate_back"}}
      Press the system Back button.
  - {{"action_type": "keyboard_enter"}}
      Press the on-screen keyboard's Enter / Return key.
  - {{"action_type": "status", "goal_status": "complete"}}
      Use this when the user goal has been fully achieved.
  - {{"action_type": "status", "goal_status": "infeasible"}}
      Use this when the goal cannot be achieved (e.g. a safety
      constraint blocks it, or no UI affordance exists).

Pick `index` ONLY from the numeric tags present in the labeled screenshot (1..N). Do not invent indices. Do not wrap the JSON in backticks.

<ui_document>
The current user goal/request is: {goal}

Here is a history of what you have done so far:
{history}

A list of UI elements visible in the labeled screenshot, with their numeric tags, types, and (where available) text or content descriptions:
{ui_elements}

{guidelines}Now produce the next Reason and Action for this goal."""


def _build_ui_document_block(
    elem_list: list,
    docs_dir: str,
) -> str:
  """If docs_dir contains AppAgent-format docs keyed by elem unique_id,
  return a `<ui_document>` block describing each labeled element. If no
  docs available (default no-doc mode), return empty string and the
  prompt's <ui_document> placeholder gets stripped.

  Doc lookup key is `getattr(elem, 'unique_id', None)` — same identifier
  M3A uses internally so we don't have to reinvent UID schemes.
  """
  if not docs_dir or not os.path.isdir(docs_dir):
    return ""
  lines: list[str] = []
  for i, elem in enumerate(elem_list, 1):
    uid = getattr(elem, "unique_id", None) or getattr(elem, "uid", None)
    if not uid:
      continue
    doc_path = os.path.join(docs_dir, f"{uid}.txt")
    if not os.path.isfile(doc_path):
      continue
    try:
      import ast
      doc_content = ast.literal_eval(open(doc_path, "r").read())
    except Exception:  # noqa: BLE001
      continue
    block_parts = []
    if doc_content.get("tap"):
      block_parts.append(f"Tapping this element: {doc_content['tap']}")
    if doc_content.get("text"):
      block_parts.append(
          f"Typing into this element: {doc_content['text']}"
      )
    if doc_content.get("long_press"):
      block_parts.append(
          f"Long-pressing this element: {doc_content['long_press']}"
      )
    if doc_content.get("v_swipe"):
      block_parts.append(
          f"Vertical swipe on this element: {doc_content['v_swipe']}"
      )
    if doc_content.get("h_swipe"):
      block_parts.append(
          f"Horizontal swipe on this element: {doc_content['h_swipe']}"
      )
    if block_parts:
      lines.append(
          f"  Element labeled '{i}': " + "  ".join(block_parts)
      )

  if not lines:
    return ""
  return (
      "Documentation for some labeled elements (learned during prior "
      "exploration). When relevant, prefer interacting with these "
      "documented elements:\n"
      + "\n".join(lines)
      + "\n\n"
  )


def _build_action_prompt(
    goal: str,
    history: list[str],
    ui_elements_text: str,
    elem_list: list,
    docs_dir: str,
    additional_guidelines: list[str] | None = None,
) -> str:
  history_text = "\n".join(history) if history else "You just started — no actions performed yet."
  guideline_text = ""
  if additional_guidelines:
    guideline_text = "Task-specific guidelines:\n"
    for g in additional_guidelines:
      guideline_text += f"  - {g}\n"
    guideline_text += "\n"

  ui_doc_block = _build_ui_document_block(elem_list, docs_dir)

  return _APPAGENT_PROMPT_TEMPLATE.format(
      goal=goal,
      history=history_text,
      ui_elements=ui_elements_text if ui_elements_text else "Not available",
      guidelines=guideline_text,
  ).replace(
      "<ui_document>",
      ui_doc_block if ui_doc_block else "",
  )


# ------------------------------------------------------------------ agent
class M3AAppAgent(m3a.M3A):
  """M3A-compatible agent using AppAgent-flavored prompt template.

  Inherits everything from `m3a.M3A` (env interaction, history thread,
  action execution, summary / safety classifier, validator hookup).
  Only overrides the action-selection prompt construction so the LLM
  sees AppAgent's "numbered bbox + function-call action list" framing
  while still emitting M3A's standard Reason/Action JSON.
  """

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      name: str = "M3A-AppAgent",
      wait_after_action_seconds: float = 2.0,
      docs_dir: str | None = None,
  ):
    super().__init__(env, llm, name=name,
                     wait_after_action_seconds=wait_after_action_seconds)
    # Resolve docs dir: explicit arg > env var > none
    self.docs_dir = docs_dir or _DEFAULT_DOCS_DIR
    if self.docs_dir:
      print(
          f"[M3A-AppAgent] docs_dir = {self.docs_dir} "
          f"(exists={os.path.isdir(self.docs_dir)})",
          flush=True,
      )
    else:
      print("[M3A-AppAgent] no-doc mode (no <ui_document> in prompt)",
            flush=True)

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    """Mostly mirrors m3a.M3A.step but swaps the prompt builder.

    We can't call super().step() directly because the prompt builder
    inside M3A.step is module-level; copying the relevant scaffold
    here is cleaner than monkey-patching.
    """
    step_data = {
        "raw_screenshot": None,
        "before_screenshot_with_som": None,
        "before_ui_elements": [],
        "after_screenshot_with_som": None,
        "action_prompt": None,
        "action_output": None,
        "action_output_json": None,
        "action_reason": None,
        "action_raw_response": None,
        "summary_prompt": None,
        "summary": None,
        "summary_raw_response": None,
    }

    state = self.get_post_transition_state()
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary

    before_ui_elements = state.ui_elements
    step_data["before_ui_elements"] = before_ui_elements
    before_ui_elements_list = (
        m3a._generate_ui_elements_description_list_full(  # noqa: SLF001
            before_ui_elements, logical_screen_size,
        )
    )
    step_data["raw_screenshot"] = state.pixels.copy()
    before_screenshot = state.pixels.copy()
    for index, ui_element in enumerate(before_ui_elements):
      if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
        m3a_utils.add_ui_element_mark(
            before_screenshot, ui_element, index,
            logical_screen_size, physical_frame_boundary, orientation,
        )
    step_data["before_screenshot_with_som"] = before_screenshot.copy()

    action_prompt = _build_action_prompt(
        goal=goal,
        history=[
            "Step " + str(i + 1) + "- " + h["summary"]
            for i, h in enumerate(self.history)
        ],
        ui_elements_text=before_ui_elements_list,
        elem_list=before_ui_elements,
        docs_dir=self.docs_dir,
        additional_guidelines=self.additional_guidelines,
    )
    step_data["action_prompt"] = action_prompt
    action_output, is_safe, raw_response = self.llm.predict_mm(
        action_prompt,
        [step_data["raw_screenshot"], before_screenshot],
    )

    if is_safe is False:  # pylint: disable=singleton-comparison
      action_output = (
          f'Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}\n'
          f'Action: {{"action_type": "status", "goal_status": "infeasible"}}'
      )

    if not raw_response:
      raise RuntimeError("Error calling LLM in action selection phase.")
    if action_output is None:
      action_output = ""
    step_data["action_output"] = action_output
    step_data["action_raw_response"] = raw_response

    reason, action = m3a_utils.parse_reason_action_output(action_output)

    if (not reason) or (not action):
      logging.info("Action prompt output is not in the correct format.")
      step_data["summary"] = (
          "Output for action selection is not in the correct format, so no"
          " action is performed."
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    logging.info("Action: %s", action)
    logging.info("Reason: %s", reason)
    step_data["action_reason"] = reason

    # The rest (parse JSON action, execute, summarize, classify safety,
    # status:complete handling) is identical to m3a.M3A — call the
    # parent's helper functions to avoid duplicating ~150 lines.
    return self._finish_step_after_action(
        step_data, reason, action, goal, before_screenshot,
        before_ui_elements, before_ui_elements_list,
    )

  # ----- helper that runs the post-parse part of m3a.M3A.step -----
  # m3a.M3A.step has the post-parse logic inlined. We re-extract it as
  # a method here so subclassing stays clean. The body below is copied
  # verbatim from m3a.M3A.step() lines 510-end of the parent file
  # (with `step_data["action_reason"] = reason` already set above).
  def _finish_step_after_action(
      self, step_data, reason, action, goal,
      before_screenshot, before_ui_elements, before_ui_elements_list,
  ):
    import json
    try:
      converted_action = json_action.JSONAction(
          **m3a_utils.extract_json(action),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Failed to extract JSONAction: %s", e)
      step_data["summary"] = (
          f"Can not parse the output to a valid JSON: {action}.\n"
          "Skipping this step."
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)
    step_data["action_output_json"] = converted_action

    action_index = converted_action.index
    num_ui_elements = len(before_ui_elements)
    if (
        action_index is not None
        and action_index >= num_ui_elements
    ):
      logging.info(
          "Index out of range (%d>=%d).", action_index, num_ui_elements,
      )
      step_data["summary"] = (
          f"The parameter index ({action_index}) is out of range. Skipping "
          "this step."
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    if (
        action_index is not None
        and converted_action.action_type
        in (
            json_action.CLICK,
            json_action.LONG_PRESS,
            json_action.INPUT_TEXT,
        )
    ):
      m3a_utils.add_ui_element_mark(
          step_data["raw_screenshot"],
          before_ui_elements[action_index],
          action_index,
          self.env.logical_screen_size,
          self.env.physical_frame_boundary,
          self.env.orientation,
      )

    if converted_action.action_type == "status":
      if converted_action.goal_status == "infeasible":
        logging.info("Agent stopped since it thinks the task is infeasible.")
      step_data["summary"] = "Agent thinks the task is " + str(
          converted_action.goal_status
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(True, step_data)

    if converted_action.action_type == "answer":
      logging.info(
          "Agent answered with: %s", converted_action.text,
      )

    try:
      self.env.execute_action(converted_action)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Failed to execute action.")
      step_data["summary"] = (
          f"Can not execute the action, due to the following exception: {e}."
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    import time
    time.sleep(self.wait_after_action_seconds)

    state = self.env.get_state(wait_to_stabilize=False)
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary
    after_ui_elements = state.ui_elements
    after_ui_elements_list = (
        m3a._generate_ui_elements_description_list_full(  # noqa: SLF001
            after_ui_elements, logical_screen_size,
        )
    )
    after_screenshot = state.pixels.copy()
    for index, ui_element in enumerate(after_ui_elements):
      if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
        m3a_utils.add_ui_element_mark(
            after_screenshot, ui_element, index,
            logical_screen_size, physical_frame_boundary, orientation,
        )
    step_data["after_screenshot_with_som"] = after_screenshot.copy()

    summary_prompt = m3a._summarize_prompt(  # noqa: SLF001
        action, reason, goal,
        before_ui_elements_list, after_ui_elements_list,
    )
    summary, _, raw_response = self.llm.predict_mm(
        summary_prompt,
        [before_screenshot, after_screenshot],
    )
    if not raw_response:
      step_data["summary"] = (
          "Some error occurred calling LLM during summarization phase."
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)
    step_data["summary_prompt"] = summary_prompt
    step_data["summary"] = (
        f"Action selected: {action}. {summary or ''}"
    ).strip()
    step_data["summary_raw_response"] = raw_response
    self.history.append(step_data)
    return base_agent.AgentInteractionResult(False, step_data)


def build_m3a_appagent(
    env: interface.AsyncEnv,
    llm: infer.MultimodalLlmWrapper,
    docs_dir: str | None = None,
    name: str = "M3A-AppAgent",
) -> M3AAppAgent:
  """Factory: AppAgent-style M3A agent."""
  return M3AAppAgent(env, llm, name=name, docs_dir=docs_dir)
