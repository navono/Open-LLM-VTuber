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
    Manages TTS tasks with controlled concurrency and sends results immediately upon completion.

    Allows a maximum number of TTS generation tasks to run concurrently.
    Results are sent via WebSocket as soon as they are ready, potentially out of order
    compared to when they were requested.
    """

    def __init__(self, max_concurrent_tasks: int = 2) -> None:
        """
        Initializes the TTSTaskManager.

        Args:
            max_concurrent_tasks: The maximum number of TTS generation tasks allowed to run simultaneously.
        """
        if max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        logger.info(
            f"Initializing TTSTaskManager with max_concurrent_tasks={max_concurrent_tasks}"
        )
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._active_tasks: set[asyncio.Task] = set()
        self._sentence_counter = 0

    async def speak(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
    ) -> None:
        """
        Queues a task to synthesize text and send it via WebSocket immediately upon completion.

        The number of concurrent TTS generations is limited by `self.max_concurrent_tasks`.
        """
        if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)) == 0:
            logger.debug(
                f"Empty TTS text from {display_text.name}, sending silent payload directly."
            )
            await self._send_silent_payload(display_text, actions, websocket_send)
            return

        logger.debug(
            f"➕ Queuing TTS task for: '''{tts_text}''' (by {display_text.name})"
        )

        tts_task = self._process_and_send_tts(
            tts_text=tts_text,
            display_text=display_text,
            actions=actions,
            tts_engine=tts_engine,
            websocket_send=websocket_send,
        )
        task = asyncio.create_task(
            tts_task, name=f"TTS_{display_text.name}_{str(uuid.uuid4())[:8]}"
        )
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        logger.debug(f"Task {task.get_name()} created and added to active tasks.")

    async def _send_silent_payload(
        self,
        display_text: DisplayText,
        actions: Optional[Actions],
        websocket_send: WebSocketSend,
    ) -> None:
        """
        Creates and sends a silent payload when there's no text to speak.
        The sentence index is incremented to maintain order.

        Args:
            display_text: The display text information
            actions: Optional actions to include with the payload
            websocket_send: Function to send the payload
        """
        try:
            payload = prepare_audio_payload(
                sentence_index=self._sentence_counter,
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            self._sentence_counter += 1
            await websocket_send(json.dumps(payload))
            logger.debug(f"Sent silent payload for {display_text.name}.")
        except Exception as e:
            logger.error(f"Error sending silent payload for {display_text.name}: {e}")

    async def _process_and_send_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
    ) -> None:
        """
        Processes TTS generation within semaphore limits and sends the result.
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
                        sentence_index=self._sentence_counter,
                        audio_stream_iterator=audio_chunk_generator,
                        display_text=display_text,
                        actions=actions,
                        forwarded=False,
                    ):
                        await websocket_send(json.dumps(audio_payload))
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
                    audio_file_path = await self._generate_audio(tts_engine, tts_text)
                    payload = prepare_audio_payload(
                        sentence_index=self._sentence_counter,
                        audio_path=audio_file_path,
                        display_text=display_text,
                        actions=actions,
                    )
                    self._sentence_counter += 1
                    logger.debug(
                        f"Task {task_name} sending payload for {display_text.name}..."
                    )
                    await websocket_send(json.dumps(payload))
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
                    display_text=display_text,
                    actions=actions,
                    websocket_send=websocket_send,
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

    async def _generate_audio(self, tts_engine: TTSInterface, text: str) -> str:
        """Generate audio file from text."""
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

    async def clear(self) -> None:
        """
        Cancel all active TTS tasks and clear the internal state.
        """
        logger.info(
            f"Clearing TTSTaskManager. Cancelling {len(self._active_tasks)} active tasks."
        )
        tasks_to_cancel = list(self._active_tasks)
        if not tasks_to_cancel:
            logger.info("No active tasks to cancel.")
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
        Waits until all currently active tasks have completed.
        This method will block until all tasks are done or until the tasks are cancelled.
        If there are no active tasks, it will return false immediately.

        Returns:
            bool: True if all tasks completed, False if no tasks were active or if some tasks are still pending for whatever reason.
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
