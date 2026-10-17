# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import signal
import time
from typing import Any, Optional
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
import requests

try:
  from google import genai as _new_genai
  from google.genai import types as _new_genai_types
  _HAS_NEW_GENAI = True
except ImportError:
  _HAS_NEW_GENAI = False

def _llm_timeout_handler(signum, frame):
  raise TimeoutError('Vertex LLM call exceeded deadline')


ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)
  return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  # Reset file pointer to start
  in_mem_file.seek(0)
  img_bytes = in_mem_file.read()
  return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling text-only LLM with a prompt.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Gemini GCP interface."""

  def __init__(
      self,
      model_name: str | None = None,
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
  ):
    if 'GCP_API_KEY' not in os.environ:
      raise RuntimeError('GCP API key not set.')

    self.model_name = model_name
    self.temperature = temperature
    self.top_p = top_p
    self.enable_safety_checks = enable_safety_checks
    self.use_vertex_express = os.environ.get('USE_VERTEX_EXPRESS') == '1'

    if self.use_vertex_express:
      if not _HAS_NEW_GENAI:
        raise RuntimeError(
            'USE_VERTEX_EXPRESS=1 requires: pip install google-genai'
        )
      # 20s default was too tight for Vertex Express from networks
      # with high latency to Google (notably mainland China without
      # a clean proxy): every Gemini call hit ReadTimeout / server
      # 504 DEADLINE_EXCEEDED, the wrapper retried up to 10x per
      # step, and each agent step ballooned to 30-60s of dead air.
      # 120s gives the server room to respond on slow links;
      # override via GCP_VERTEX_TIMEOUT_MS for tighter ceilings.
      _vertex_timeout_ms = int(os.environ.get(
          'GCP_VERTEX_TIMEOUT_MS', '120000'
      ))
      self._vertex_client = _new_genai.Client(
          vertexai=True,
          api_key=os.environ['GCP_API_KEY'],
          http_options=_new_genai_types.HttpOptions(
              timeout=_vertex_timeout_ms,
          ),
      )
      self.llm = None
    else:
      genai.configure(api_key=os.environ['GCP_API_KEY'])
      self.llm = genai.GenerativeModel(
          model_name,
          safety_settings=None
          if enable_safety_checks
          else SAFETY_SETTINGS_BLOCK_NONE,
          generation_config=generation_types.GenerationConfig(
              temperature=temperature, top_p=top_p
          ),
      )
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)

  def predict(
      self,
      text_prompt: str,
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(
        text_prompt, [], enable_safety_checks, generation_config
    )

  def is_safe(self, raw_response):
    try:
      return (
          raw_response.candidates[0].finish_reason
          != answer_types.FinishReason.SAFETY
      )
    except Exception:  # pylint: disable=broad-exception-caught
      #  Assume safe if the response is None or doesn't have candidates.
      return True

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    if self.use_vertex_express:
      return self._predict_mm_vertex_express(text_prompt, images)
    counter = self.max_retry
    retry_delay = 1.0
    output = None
    while counter > 0:
      try:
        output = self.llm.generate_content(
            [text_prompt] + [Image.fromarray(image) for image in images],
            safety_settings=None
            if enable_safety_checks
            else SAFETY_SETTINGS_BLOCK_NONE,
            generation_config=generation_config,
        )
        # output.text raises if the response has no Parts (e.g. empty
        # content with finish_reason=STOP).  That is not a transient
        # error — retrying won't help — so catch it separately.
        try:
          text = output.text
        except ValueError:
          return '', True, output
        return text, True, output
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print(f'Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2

    if (output is not None) and (not self.is_safe(output)):
      return ERROR_CALLING_LLM, False, output
    return ERROR_CALLING_LLM, None, None

  def _predict_mm_vertex_express(
      self,
      text_prompt: str,
      images: list[np.ndarray],
  ) -> tuple[str, Optional[bool], Any]:
    pil_images = [Image.fromarray(img) for img in images]
    contents = [text_prompt] + pil_images

    config = _new_genai_types.GenerateContentConfig(
        temperature=self.temperature,
        top_p=self.top_p,
    )

    TIMEOUT_SEC = 25
    counter = max(self.max_retry, 5)
    retry_delay = 0.5
    output = None
    while counter > 0:
      signal.signal(signal.SIGALRM, _llm_timeout_handler)
      signal.alarm(TIMEOUT_SEC)
      try:
        output = self._vertex_client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        signal.alarm(0)
        try:
          text = output.text or ''
        except Exception:  # pylint: disable=broad-exception-caught
          text = ''
        return text, True, output
      except TimeoutError:
        signal.alarm(0)
        counter -= 1
        print(
            f'[Vertex TIMEOUT >{TIMEOUT_SEC}s] retry in {retry_delay}s'
            f' ({counter} left)'
        )
        if counter > 0:
          time.sleep(retry_delay)
          retry_delay *= 1.5
      except Exception as e:  # pylint: disable=broad-exception-caught
        signal.alarm(0)
        counter -= 1
        print(f'Vertex error: {type(e).__name__}: {str(e)[:200]}')
        if counter > 0:
          time.sleep(retry_delay)
          retry_delay *= 1.5
    return ERROR_CALLING_LLM, None, output

  def generate(
      self,
      contents: (
          content_types.ContentsType | list[str | np.ndarray | Image.Image]
      ),
      safety_settings: safety_types.SafetySettingOptions | None = None,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Any]:
    """Exposes the generate_content API.

    Args:
      contents: The input to the LLM.
      safety_settings: Safety settings.
      generation_config: Generation config.

    Returns:
      The output text and the raw response.
    Raises:
      RuntimeError:
    """
    counter = self.max_retry
    retry_delay = 1.0
    response = None
    if isinstance(contents, list):
      contents = self.convert_content(contents)
    while counter > 0:
      try:
        response = self.llm.generate_content(
            contents=contents,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        return response.text, response
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2
    raise RuntimeError(f'Error calling LLM. {response}.')

  def convert_content(
      self,
      contents: list[str | np.ndarray | Image.Image],
  ) -> content_types.ContentsType:
    """Converts a list of contents to a ContentsType."""
    converted = []
    for item in contents:
      if isinstance(item, str):
        converted.append(item)
      elif isinstance(item, np.ndarray):
        converted.append(Image.fromarray(item))
      elif isinstance(item, Image.Image):
        converted.append(item)
    return converted


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI GPT4 wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
      # Default sized for thinking-mode models (Kimi K2.5, GPT-5,
      # Qwen3.5-thinking) which routinely burn 4-8K tokens on internal
      # reasoning before emitting "Reason:/Action:". Smaller models
      # never exceed a few hundred tokens, and max_tokens is a CEILING
      # not a charge — providers bill for actual output tokens, so
      # raising the default is free for non-thinking models and saves
      # one wasted "4096 -> length -> bump to 8192" round-trip per
      # step for thinking models. Override per-instance if needed.
      max_tokens: int = 8192,
  ):
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self.openai_api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.model = model_name
    self.max_tokens = max_tokens
    # Hard cap on the auto-doubling escalation when finish_reason="length"
    # is hit with empty content. Default 8192 -> bump to 16384 once ->
    # give up. 16384 matches Kimi K2.5's HF model card recommendation
    # (out_seq_length=16384) — covers the rare task where 8192 is still
    # not enough for the reasoning chain, but stops short of unbounded
    # doubling.
    self._max_tokens_cap = 16384
    # OpenAI-compatible providers (OpenRouter, Azure-OAI, vLLM, ...) all
    # accept the same /v1/chat/completions payload. Override the endpoint
    # base via OPENAI_BASE_URL when targeting a non-default provider; the
    # default preserves the original https://api.openai.com behavior so
    # existing setups keep working unchanged.
    base_url = os.environ.get(
        'OPENAI_BASE_URL', 'https://api.openai.com/v1'
    ).rstrip('/')
    self.chat_completions_url = f'{base_url}/chat/completions'

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openai_api_key}',
    }

    # max_tokens may be bumped within the retry loop when the API reports
    # finish_reason="length" with empty content (long-thinking model burned
    # the budget on reasoning tokens before emitting Reason:/Action:).
    current_max_tokens = self.max_tokens
    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_tokens': current_max_tokens,
    }

    # Gpt-4v supports multiple images, just need to insert them in the content
    # list.
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        # (connect_timeout, read_timeout). Without an explicit timeout
        # `requests.post` blocks forever on half-open TCP connections —
        # observed on OpenRouter when the upstream provider stalls
        # mid-response (no exception, no retry, the whole task hangs).
        # 180s read covers the slowest reasoning-mode models (Kimi K2.5,
        # qwen3.5-122b at high load) without false-positive aborts.
        response = requests.post(
            self.chat_completions_url,
            headers=headers,
            json=payload,
            timeout=(10, 180),
        )
        if response.ok:
          # Some providers (OpenRouter routes for reasoning models, certain
          # Qwen / Kimi backends, content-filter trips) return HTTP 200 with
          # `choices[0].message.content == null`. The answer, if any, lives
          # in `reasoning_content`. Treating null content as success caused
          # downstream m3a regex to crash with "expected string or bytes-
          # like object, got 'NoneType'" and aborted the entire task.
          try:
            body = response.json()
          except Exception as je:  # noqa: BLE001
            body = None
            print(
                f'[Gpt4Wrapper] non-JSON 200 body '
                f'(len={len(response.text)}): {response.text[:300]!r} '
                f'-- {type(je).__name__}: {je}'
            )
          if body and 'choices' in body and body['choices']:
            msg = (body['choices'][0] or {}).get('message') or {}
            content = msg.get('content')
            if content is None or content == '':
              content = msg.get('reasoning_content')
            finish_reason = body['choices'][0].get('finish_reason')
            if isinstance(content, str) and content:
              return content, None, response
            # 200 OK but no usable text. finish_reason="length" is NOT
            # transient — same payload + same max_tokens will produce
            # the same truncated output. Bump the budget ONCE; if still
            # empty after the bump, give up to stop wasting API spend.
            if finish_reason == 'length' and current_max_tokens < self._max_tokens_cap:
              new_budget = min(current_max_tokens * 2, self._max_tokens_cap)
              print(
                  f'[Gpt4Wrapper] finish_reason=length, content empty. '
                  f'Bumping max_tokens {current_max_tokens} -> {new_budget} '
                  f'and retrying once.'
              )
              current_max_tokens = new_budget
              payload['max_tokens'] = current_max_tokens
              # Do not decrement counter on the budget bump (this is a
              # *different* call), but do not back off either.
              continue
            if finish_reason == 'length':
              # Already at cap and still nothing usable — abort retries.
              print(
                  f'[Gpt4Wrapper] finish_reason=length at max_tokens='
                  f'{current_max_tokens} (cap). Giving up to avoid '
                  f'further API spend.'
              )
              return ERROR_CALLING_LLM, None, response
            print(
                f'[Gpt4Wrapper] empty content from API '
                f'(finish_reason={finish_reason!r}, '
                f'has_reasoning={bool(msg.get("reasoning_content"))}, '
                f'has_tool_calls={bool(msg.get("tool_calls"))}). Retrying.'
            )
            time.sleep(wait_seconds)
            wait_seconds *= 2
            counter -= 1
            continue
        # Non-2xx OR malformed JSON OR no choices -> log + back off
        try:
          err = response.json()
          err_msg = (
              err.get('error', {}).get('message')
              if isinstance(err.get('error'), dict)
              else str(err.get('error', err))
          )
        except Exception:  # noqa: BLE001
          err_msg = response.text[:300]
        print(
            f'[Gpt4Wrapper] API error status={response.status_code}: {err_msg}'
        )
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
      except (requests.exceptions.SSLError,
              requests.exceptions.ConnectionError,
              requests.exceptions.Timeout,
              requests.exceptions.ChunkedEncodingError) as e:
        # Transient network-class errors (OpenRouter connection resets,
        # SSL EOF, read timeouts). These almost always clear within a
        # second or two — the original 20s/40s/80s exponential backoff
        # was tuned for rate-limit retries and wastes wall time here.
        time.sleep(2)
        counter -= 1
        print(f'[Gpt4Wrapper] network blip ({type(e).__name__}); '
              f'retrying in 2s. {str(e)[:120]}')
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Other / unknown errors — keep the original conservative backoff.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None
