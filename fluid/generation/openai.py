# fluid/generation/openai.py

import logging
from threading import Event, Lock
from typing import Optional
import os
from openai import OpenAI
from fluid.utill import generate_buffer_id, _build_logger

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "openai_comaptible.log")
logger = _build_logger(log_dir=LOG_DIR, log_file=LOG_FILE)


class OpenAICom:
    def __init__(self):
        # A lock is needed because multiple threads could call .openai() or
        # .stop() at the same time, and dict access isn't guaranteed safe
        # under concurrent writes the way list.append() is.
        self._lock = Lock()

        # Per-stream storage, keyed by buffer_id, instead of one shared
        # list/flag for every call. This means two streams running at the
        # same time can't interfere with each other.
        self._buffers_by_id: dict[str, list] = {}
        self._stop_events_by_id: dict[str, Event] = {}

        # Kept for backward compatibility: setting THIS stops every
        # currently running stream, same as the original behavior. Prefer
        # `self.stop(buffer_id)` for stopping just one stream.
        self.stop_event = Event()

    # -----------------------------------------------------------------
    # Public helpers for managing streams from outside the class
    # -----------------------------------------------------------------
    def stop(self, buffer_id: str) -> None:
        """Signals just ONE stream (by its buffer_id) to stop, without affecting others."""
        with self._lock:
            event = self._stop_events_by_id.get(buffer_id)
        if event is not None:
            event.set()
        else:
            logger.warning(f"stop() called with unknown buffer_id={buffer_id!r}")

    def get_buffer(self, buffer_id: str) -> list:
        """Returns the chunks collected so far for a given stream. Empty list if unknown."""
        with self._lock:
            return list(self._buffers_by_id.get(buffer_id, []))

    # -----------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------
    def _validate_chat_inputs(
        self, messages, base_url, api_key, model, tools, response_format
    ):
        if not messages or not isinstance(messages, list):
            raise ValueError(f"'messages' must be a non-empty list, got: {messages!r}")

        if not base_url or not isinstance(base_url, str):
            raise ValueError(
                f"'base_url' must be a non-empty string, got: {base_url!r}"
            )

        if not api_key or not isinstance(api_key, str):
            raise ValueError(f"'api_key' must be a non-empty string, got: {api_key!r}")

        if not model or not isinstance(model, str):
            raise ValueError(f"'model' must be a non-empty string, got: {model!r}")

        if tools is not None and not isinstance(tools, list):
            raise ValueError(f"'tools' must be a list or None, got: {tools!r}")

        if response_format is not None and not isinstance(response_format, dict):
            raise ValueError(
                f"'response_format' must be a dict or None, got: {response_format!r}"
            )

    def _validate_emb_inputs(self, model, input_, base_url, api_key):
        if not model or not isinstance(model, str):
            raise ValueError(f"'model' must be a non-empty string, got: {model!r}")

        if not input_ or not isinstance(input_, (str, list)):
            raise ValueError(
                f"'input' must be a non-empty string or list, got: {input_!r}"
            )

        if not base_url or not isinstance(base_url, str):
            raise ValueError(
                f"'base_url' must be a non-empty string, got: {base_url!r}"
            )

        if not api_key or not isinstance(api_key, str):
            raise ValueError(f"'api_key' must be a non-empty string, got: {api_key!r}")

    # -----------------------------------------------------------------
    # Chat streaming
    # -----------------------------------------------------------------
    def openai(
        self,
        messages,
        base_url,
        api_key,
        model,
        tools=None,
        tool_choice="auto",
        parallel_tool_calls=False,
        timeout=None,
        max_retries=0,
        http_client=None,
        default_headers=None,
        response_format=None,  # was {"type": "text"} -- fixed mutable-default bug, see notes above
        max_completion_tokens=None,
        temperature=None,
        top_p=None,
        presence_penalty=None,
        frequency_penalty=None,
    ):
        """
        Streams a chat completion. Returns a generator you iterate with:
            for chunk in com.openai(...):
                ...

        Bad input (missing/wrong-type required args) raises immediately,
        BEFORE you start iterating -- this is why validation happens here,
        in a normal method, rather than inside the generator itself (code
        inside a generator function doesn't run until you start pulling
        values from it, which would otherwise delay the error and make it
        confusing to debug).

        Once the stream is running, operational failures (bad API key,
        network drop, model not found, etc.) do NOT raise -- instead you
        get one final chunk shaped like:
            {"buffer_id": ..., "chunk": None, "error": "..."}
        Check `"error" in chunk` in your loop to detect this.
        """
        self._validate_chat_inputs(
            messages, base_url, api_key, model, tools, response_format
        )

        if response_format is None:
            response_format = {"type": "text"}

        with self._lock:
            buffer_id = generate_buffer_id(
                buffers=list(self._buffers_by_id.keys()), length=8
            )
            self._buffers_by_id[buffer_id] = []
            self._stop_events_by_id[buffer_id] = Event()

        return self._stream_chat(
            buffer_id=buffer_id,
            messages=messages,
            base_url=base_url,
            api_key=api_key,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            default_headers=default_headers,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

    def _stream_chat(
        self,
        buffer_id,
        messages,
        base_url,
        api_key,
        model,
        tools,
        tool_choice,
        parallel_tool_calls,
        timeout,
        max_retries,
        http_client,
        default_headers,
        response_format,
        max_completion_tokens,
        temperature,
        top_p,
        presence_penalty,
        frequency_penalty,
    ):
        logger.info(
            f"[chat] START buffer_id={buffer_id} model={model!r} base_url={base_url!r}"
        )

        try:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                http_client=http_client,
                default_headers=default_headers,
            )
        except Exception as e:
            logger.error(
                f"[chat] Failed to build OpenAI client for buffer_id={buffer_id}: {e}",
                exc_info=True,
            )
            yield {
                "buffer_id": buffer_id,
                "chunk": None,
                "error": f"client setup failed: {e}",
            }
            return

        # tool_choice / parallel_tool_calls are only meaningful (and, on some
        # OpenAI-compatible backends, only ACCEPTED) when tools are actually
        # provided -- so we only include them in the request when tools is set.
        create_kwargs = dict(
            messages=messages,
            model=model,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stream=True,
        )
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = tool_choice
            create_kwargs["parallel_tool_calls"] = parallel_tool_calls

        stream = None
        chunk_count = 0

        try:
            stream = client.chat.completions.create(**create_kwargs)

            for _chunk in stream:
                chunk = {"buffer_id": buffer_id, "chunk": _chunk}

                with self._lock:
                    self._buffers_by_id[buffer_id].append(chunk)
                    stop_now = (
                        self.stop_event.is_set()
                        or self._stop_events_by_id[buffer_id].is_set()
                    )

                yield chunk
                chunk_count += 1

                if stop_now:
                    logger.info(
                        f"[chat] STOPPED early buffer_id={buffer_id} after {chunk_count} chunk(s)"
                    )
                    break

            else:
                logger.info(f"[chat] DONE buffer_id={buffer_id} chunks={chunk_count}")

        except Exception as e:
            logger.error(
                f"[chat] FAILED buffer_id={buffer_id} after {chunk_count} chunk(s): {e}",
                exc_info=True,
            )
            yield {"buffer_id": buffer_id, "chunk": None, "error": str(e)}

        finally:
            # guaranteed to run even if the loop above raised or broke early
            if stream is not None:
                try:
                    stream.close()
                except Exception as e:
                    logger.warning(
                        f"[chat] Failed to close stream for buffer_id={buffer_id}: {e}"
                    )

    # -----------------------------------------------------------------
    # Embeddings
    # -----------------------------------------------------------------
    def openai_emb(
        self,
        model,
        input,
        base_url,
        api_key,
        timeout=None,
        max_retries=0,
        http_client=None,
        default_headers=None,
    ):
        """
        Returns the embeddings response on success, same shape as before.
        On failure, logs the error and returns None instead of raising --
        check `if response is None:` in your calling code.
        """
        self._validate_emb_inputs(model, input, base_url, api_key)

        logger.info(f"[emb] START model={model!r} base_url={base_url!r}")

        try:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                http_client=http_client,
                default_headers=default_headers,
            )

            response = client.embeddings.create(model=model, input=input)

            logger.info(f"[emb] DONE model={model!r}")
            return response

        except Exception as e:
            logger.error(f"[emb] FAILED model={model!r}: {e}", exc_info=True)
            return None

