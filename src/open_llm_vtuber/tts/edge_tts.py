import os
import sys
from typing import AsyncGenerator

import edge_tts
from loguru import logger
from .tts_interface import TTSInterface

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# Check out doc at https://github.com/rany2/edge-tts
# Use `edge-tts --list-voices` to list all available voices


class TTSEngine(TTSInterface):
    def __init__(self, voice="en-US-AvaMultilingualNeural", stream=False):
        """
        Initializes edge tts instance.
        Args:
            voice (str): The voice to use for TTS.
            stream (bool): Whether to stream the audio or not.
        """
        super().__init__(stream=stream)

        if stream:
            logger.critical("Edge-TTS 流式输出启动！")
        logger.warning(f"stream: {self.stream}")

        self.voice = voice

        self.temp_audio_file = "temp"
        self.file_extension = "mp3"
        self.new_audio_dir = "cache"

        if not os.path.exists(self.new_audio_dir):
            os.makedirs(self.new_audio_dir)

    def generate_audio(self, text, file_name_no_ext=None):
        """
        Generate speech audio file using TTS.
        text: str
            the text to speak
        file_name_no_ext: str
            name of the file without extension


        Returns:
        str: the path to the generated audio file

        """
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        try:
            communicate = edge_tts.Communicate(text, self.voice)
            communicate.save_sync(file_name)
        except Exception as e:
            logger.critical(f"\nError: edge-tts unable to generate audio: {e}")
            logger.critical("It's possible that edge-tts is blocked in your region.")
            return None

        return file_name

    async def async_stream_audio(self, sentence_text: str) -> AsyncGenerator[bytes, None]:
        """
        Async generator to stream audio data from edge-tts for a given text.
        Args:
            sentence_text (str): The text to be converted to speech.
        Yields:
            bytes: The audio data chunks.
        """
        try:
            communicate = edge_tts.Communicate(sentence_text, self.voice)

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
                #
                # elif chunk["type"] == "WordBoundary":
                # submaker.feed(chunk)
        except edge_tts.exceptions.NoAudioReceived as e:
            logger.error(f"No audio received from edge-tts: {e}")
            raise e
        except edge_tts.exceptions.UnexpectedResponse as e:
            logger.error(f"Unexpected response from edge-tts: {e}")
            raise e
        except edge_tts.exceptions.UnknownResponse as e:
            logger.error(f"Unknown response from edge-tts: {e}")
            raise e
        except edge_tts.exceptions.WebSocketError as e:
            logger.error(f"WebSocket error with edge-tts: {e}")
            raise e


# en-US-AvaMultilingualNeural
# en-US-EmmaMultilingualNeural
# en-US-JennyNeural
