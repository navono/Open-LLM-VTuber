import re
import json
import uuid
import asyncio
from datetime import datetime
from typing import Optional
from collections.abc import AsyncGenerator

from loguru import logger

from ..agent.output_types import DisplayText, Actions
from ..tts.tts_interface import TTSInterface
from ..utils.audio_payload import prepare_audio_payload
from ..utils.audio_stream_payload import stream_audio_payload
from .types import WebSocketSend


class TTSTaskManager:
    """
    Manages Text-to-Speech (TTS) generation, payload preparation, and delivery tasks.

    This class handles the lifecycle of TTS generation for a single AI response,
    controlling concurrency and managing the sending of audio payloads (either
    as complete files or streamed chunks) via a WebSocket connection.

    A single instance of `TTSTaskManager` is intended to manage one complete
    response from the AI, identified by a unique `response_uid`. Each sentence
    within the response is processed individually. If streaming is enabled,
    audio chunks for each sentence are sent as sub-sentence payloads.

    Concurrency is managed using an `asyncio.Semaphore` to limit the number of
    simultaneous TTS generation tasks.

    Attributes:
        websocket_send (WebSocketSend): The function used to send data over the WebSocket.
        max_concurrent_tasks (int): Maximum number of concurrent TTS tasks allowed.
        response_uid (Optional[str]): Unique identifier for the current AI response.
                                      Generated when the first sentence is processed.
    """

    def __init__(
        self, websocket_send: WebSocketSend, max_concurrent_tasks: int = 2
    ) -> None:
        """
        Initializes the TTSTaskManager.

        Args:
            websocket_send: A callable function responsible for sending data
                            through the WebSocket connection.
            max_concurrent_tasks: The maximum number of TTS generation tasks
                                  allowed to run simultaneously. Must be positive.

        Raises:
            ValueError: If `max_concurrent_tasks` is not positive.
            TypeError: If `websocket_send` is not a callable.
        """
        if max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        logger.info(
            f"Initializing TTSTaskManager with max_concurrent_tasks={max_concurrent_tasks}"
        )
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._active_tasks: set[asyncio.Task] = set()
        self.websocket_send = websocket_send

        if not websocket_send:
            logger.critical(
                f"websocket_send must be a callable of type WebSocketSend, got {type(websocket_send)}"
            )
            raise TypeError(
                f"websocket_send must be a callable of type WebSocketSend, got {type(websocket_send)}"
            )

        # will be set when the response is started
        self.response_uid = None  # iso 8601 + uuid4
        logger.info("DONE")

    async def speak(
        self,
        sentence_index: int,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        tts_engine: TTSInterface,
    ) -> None:
        """
        Queues a task to synthesize text and send the resulting audio via WebSocket.

        If `tts_text` is effectively empty after removing whitespace and punctuation,
        a silent payload is sent immediately. Otherwise, a TTS generation task is
        created and managed by the semaphore. The audio payload (streamed or complete)
        is sent as soon as it's ready.

        This method automatically calls `start_response` if it hasn't been called yet
        for the current response.

        Args:
            sentence_index: The zero-based index of the sentence within the response.
            tts_text: The text content to be synthesized.
            display_text: Information about how the text should be displayed.
            actions: Optional actions associated with this sentence.
            tts_engine: The TTS engine instance to use for synthesis.
        """

        if not self.response_uid:
            await self.start_response()

        if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)) == 0:
            logger.debug(
                f"Empty TTS text from {display_text.name}, sending silent payload directly."
            )
            await self._send_silent_payload(
                sentence_index=sentence_index,
                display_text=display_text,
                actions=actions,
            )
            return

        logger.debug(
            f"➕ Queuing TTS task for: '''{tts_text}''' (by {display_text.name})"
        )

        tts_task = self._process_and_send_tts(
            sentence_index=sentence_index,
            tts_text=tts_text,
            display_text=display_text,
            actions=actions,
            tts_engine=tts_engine,
        )
        task = asyncio.create_task(
            tts_task, name=f"TTS_{display_text.name}_{str(uuid.uuid4())[:8]}"
        )
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        logger.debug(f"Task {task.get_name()} created and added to active tasks.")

    async def _send_silent_payload(
        self,
        sentence_index: int,
        display_text: DisplayText,
        actions: Optional[Actions],
    ) -> None:
        """
        Creates and sends a silent audio payload via WebSocket.

        Used when the input text for TTS is empty or effectively silent.

        Args:
            sentence_index: The index of the sentence this silent payload corresponds to.
            display_text: The display text information.
            actions: Optional actions to include with the payload.
        """
        try:
            payload = prepare_audio_payload(
                response_uid=self.response_uid,
                sentence_index=sentence_index,
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            await self.websocket_send(json.dumps(payload))
            logger.debug(f"Sent silent payload for {display_text.name}.")
        except Exception as e:
            logger.error(f"Error sending silent payload for {display_text.name}: {e}")

    async def _process_and_send_tts(
        self,
        sentence_index: int,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        tts_engine: TTSInterface,
    ) -> None:
        """
        Generates TTS audio and sends it via WebSocket, respecting concurrency limits.

        Acquires the semaphore before proceeding with TTS generation. If the
        `tts_engine` supports streaming, it streams audio chunks. Otherwise, it
        generates a complete audio file and sends it. Handles potential errors
        during generation or sending, attempting to send a silent fallback payload
        on failure. Ensures temporary audio files are cleaned up.

        Args:
            sentence_index: The index of the sentence being processed.
            tts_text: The text to synthesize.
            display_text: Display information associated with the text.
            actions: Optional actions associated with the text.
            tts_engine: The TTS engine instance.

        Raises:
            asyncio.CancelledError: If the task is cancelled externally.
            Exception: Catches and logs other exceptions during processing,
                       attempting to send a silent payload as fallback.
        """
        audio_file_path: Optional[str] = None
        task_name = asyncio.current_task().get_name()
        try:
            logger.debug(f"Task {task_name} waiting to acquire semaphore...")
            async with self._semaphore:
                if tts_engine.stream:
                    logger.info("streaming tts is here")
                    audio_chunk_generator: AsyncGenerator[bytes, None] = (
                        tts_engine.async_stream_audio(sentence_text=tts_text)
                    )

                    logger.debug(
                        f"Task {task_name} sending streaming payload for {display_text.name}, tts_text: {tts_text}"
                    )
                    async for audio_payload in stream_audio_payload(
                        audio_stream_iterator=audio_chunk_generator,
                        response_uid=self.response_uid,
                        sentence_index=sentence_index,
                        display_text=display_text,
                        actions=actions,
                        forwarded=False,
                    ):
                        await self.websocket_send(json.dumps(audio_payload))
                        logger.debug(
                            f"✅ Task {task_name} successfully sent streaming payload."
                        )
                    logger.debug(
                        f"Task {task_name} finished sending streaming payload for {display_text.name}."
                    )

                else:  # TTS streaming is not activated
                    logger.debug(
                        f"Task {task_name} acquired semaphore. Processing TTS for '''{tts_text}'''..."
                    )
                    audio_file_path = await self._generate_audio_file(
                        tts_engine, tts_text
                    )
                    payload = prepare_audio_payload(
                        sentence_index=sentence_index,
                        audio_path=audio_file_path,
                        display_text=display_text,
                        actions=actions,
                    )
                    logger.debug(
                        f"Task {task_name} sending payload for {display_text.name}..."
                    )
                    await self.websocket_send(json.dumps(payload))
                    logger.info(
                        f"✅ Task {task_name} successfully sent payload for '{display_text.text}' by {display_text.name}."
                    )
        except asyncio.CancelledError:
            logger.warning(f"Task {task_name} was cancelled.")
        except Exception as e:
            logger.error(
                f"Task {task_name} failed during TTS processing or sending for '{display_text.text}': {e}",
                exc_info=True,
            )
            try:
                logger.warning(
                    f"Task {task_name} sending silent fallback payload for '{display_text.text}'."
                )
                await self._send_silent_payload(
                    sentence_index=sentence_index,
                    display_text=display_text,
                    actions=actions,
                )
                logger.info(
                    f"Task {task_name} sent silent fallback payload for '{display_text.text}'."
                )
            except Exception as send_err:
                logger.error(
                    f"Task {task_name} failed even to send the fallback (silent) payload: {send_err}"
                )
        finally:
            if audio_file_path:
                try:
                    tts_engine.remove_file(audio_file_path)
                    logger.debug(
                        f"Task {task_name} cleaned up audio file: {audio_file_path}"
                    )
                except Exception as cleanup_err:
                    logger.error(
                        f"Task {task_name} failed to clean up audio file {audio_file_path}: {cleanup_err}"
                    )

    async def _generate_audio_file(self, tts_engine: TTSInterface, text: str) -> str:
        """
        Generates a complete audio file from text using the provided TTS engine.

        This method is used when the TTS engine does not support streaming.

        Args:
            tts_engine: The TTS engine instance.
            text: The text to synthesize.

        Returns:
            The file path to the generated audio file.

        Raises:
            Exception: Propagates exceptions raised by the TTS engine during generation.
        """
        logger.debug(f"🏃 Generating audio for '''{text}'''...")
        file_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        )
        try:
            audio_path = await tts_engine.async_generate_audio(
                text=text,
                file_name_no_ext=file_name,
            )
            return audio_path
        except Exception as e:
            logger.error(f"TTS engine failed to generate audio for '{text}': {e}")
            raise

    async def start_response(self) -> str:
        """
        Initializes a new response sequence and sends a start control message.

        Generates a unique `response_uid` for the response, stores it in the
        instance, and sends a 'start-of-response' control message via WebSocket.
        This is typically called automatically by the first `speak` call for a
        new response.

        Returns:
            The generated unique response UID.
        """
        response_uid = f"{datetime.now().isoformat()}_{str(uuid.uuid4())[:8]}"
        logger.info(f"Response start. UID set to {response_uid}")
        await self.websocket_send(
            json.dumps(
                {
                    "type": "control",
                    "text": "start-of-response",
                    "response_id": response_uid,
                }
            )
        )
        self.response_uid = response_uid
        return self.response_uid

    async def end_response(self, sentence_index: int) -> None:
        """
        Sends an end-of-response control message via WebSocket.

        Indicates that all sentences for the current `response_uid` have been processed.

        Args:
            sentence_index: The index intended for the *next* sentence, effectively
                            indicating the total number of sentences processed (0-based index + 1).
        """
        await self.websocket_send(
            json.dumps(
                {
                    "type": "control",
                    "text": "end-of-response",
                    "response_id": self.response_uid,
                    "sentence_index": sentence_index,
                }
            )
        )
        logger.info("End of response sent.")
        logger.info(f"Response UID is {self.response_uid}")

    async def clear(self) -> None:
        """
        Cancels all active TTS tasks and resets the manager's state for a new response.

        Iterates through all currently running TTS tasks managed by this instance,
        requests their cancellation, and waits for them to finish cancelling.
        Clears the internal set of active tasks. This should be called when a
        response is definitively finished or interrupted to clean up resources.
        """

        logger.info(
            f"Clearing TTSTaskManager. Cancelling {len(self._active_tasks)} active tasks."
        )
        tasks_to_cancel = list(self._active_tasks)
        if not tasks_to_cancel:
            logger.info("No active tasks to cancel. Great!")
            logger.info("TTSTaskManager cleared.")
            return
        for task in tasks_to_cancel:
            task.cancel()
            logger.debug(f"Requested cancellation for task {task.get_name()}")
        try:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            logger.info("Finished waiting for tasks cancellation process.")
        except asyncio.CancelledError:
            logger.info("Gather was cancelled (expected if clear itself is cancelled).")
        if self._active_tasks:
            logger.warning(
                f"Some tasks might not have been removed from the set after cancellation: {self._active_tasks}"
            )
            self._active_tasks.clear()
        logger.info("TTSTaskManager cleared.")

    async def wait_for_completion(self) -> bool:
        """
        Waits until all currently managed TTS tasks have completed.

        Blocks execution until all tasks in the `_active_tasks` set finish.
        Useful for ensuring all audio generation and sending is complete before
        proceeding with subsequent actions related to the response.

        Returns:
            True if all tasks completed successfully, False if there were no tasks
            to wait for or if any tasks remained pending after waiting (which
            might indicate an issue or timeout, though timeout is not explicitly handled here).
        """
        if not self._active_tasks:
            logger.info("No active tasks to wait for.")
            return False
        logger.info(
            f"Waiting for {len(self._active_tasks)} active tasks to complete..."
        )
        tasks_to_wait_for = list(self._active_tasks)
        done, pending = await asyncio.wait(
            tasks_to_wait_for, return_when=asyncio.ALL_COMPLETED
        )
        if pending:
            #!TODO We don't have a proper timeout management logic
            #!TODO for now, we just log the pending tasks
            logger.critical(f"{len(pending)} tasks still pending.")
            logger.debug(f"Pending tasks: {[task.get_name() for task in pending]}")
            return False
        else:
            logger.info("All waited-for tasks have completed.")
            return True
