"""
Module for streaming audio payload processing.

This module provides functionality to process asynchronous streams of audio chunks
(MP3 or WAV) and yield json payloads ready to be sent to the client.

Look at the `stream_audio_payload` function for the main functionality.
"""

import os
import io
import json
import base64
import struct
import asyncio
from datetime import datetime
from collections import deque
from typing import (
    AsyncIterator,
    AsyncGenerator,
    Deque,
    Tuple,
    Optional,
    List,
    Dict,
    Any,
    Union,
)

from loguru import logger
from pydub import AudioSegment
from pydub.utils import make_chunks

from ..agent.output_types import Actions, DisplayText


# --- Constants for MP3 Frame Parsing ---
# Samples per frame based on MPEG version and layer
SAMPLES_PER_FRAME: Dict[Tuple[Union[int, float], int], int] = {
    (1, 1): 384,  # MPEG Version 1, Layer 1
    (1, 2): 1152,  # MPEG Version 1, Layer 2
    (1, 3): 1152,  # MPEG Version 1, Layer 3 (MP3)
    (2, 1): 384,  # MPEG Version 2, Layer 1
    (2, 2): 1152,  # MPEG Version 2, Layer 2
    (2, 3): 576,  # MPEG Version 2, Layer 3
    (2.5, 1): 384,  # MPEG Version 2.5, Layer 1
    (2.5, 2): 1152,  # MPEG Version 2.5, Layer 2
    (2.5, 3): 576,  # MPEG Version 2.5, Layer 3
}

# Sample rates in Hz based on MPEG version and index
SAMPLE_RATES: Dict[Union[int, float], Tuple[int, int, int]] = {
    1: (44100, 48000, 32000),  # MPEG Version 1
    2: (22050, 24000, 16000),  # MPEG Version 2
    2.5: (11025, 12000, 8000),  # MPEG Version 2.5
}

# Bitrates in kbps based on MPEG version, layer, and index
BITRATES: Dict[Tuple[Union[int, float], int], Tuple[int, ...]] = {
    # MPEG Version 1: (Layer 1, Layer 2, Layer 3)
    (1, 1): (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
    (1, 2): (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
    (1, 3): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    # MPEG Version 2 & 2.5: lower bitrates
    (2, 1): (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
    (2, 2): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
BITRATES[(2.5, 1)] = BITRATES[(2, 1)]
BITRATES[(2, 3)] = BITRATES[(2, 2)]
BITRATES[(2.5, 2)] = BITRATES[(2, 2)]
BITRATES[(2.5, 3)] = BITRATES[(2, 2)]


def _create_wav_header(
    sample_rate: int, channels: int, sample_width: int, num_frames: int
) -> bytes:
    """
    Create a WAV header for PCM data.

    Args:
        sample_rate (int): Samples per second (Hz).
        channels (int): Number of audio channels.
        sample_width (int): Bytes per sample (e.g., 2 for 16-bit audio).
        num_frames (int): Number of audio frames (samples per channel).

    Returns:
        bytes: A valid WAV header for the given PCM data properties.

    Raises:
        ValueError: If resulting data size is negative.
    """
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    subchunk2_size = num_frames * block_align

    if subchunk2_size < 0:
        raise ValueError("Number of frames results in negative data size.")

    header = io.BytesIO()
    # RIFF chunk descriptor
    header.write(b"RIFF")
    header.write(struct.pack("<I", 36 + subchunk2_size))  # ChunkSize
    header.write(b"WAVE")
    # fmt subchunk
    header.write(b"fmt ")
    header.write(struct.pack("<I", 16))  # Subchunk1Size for PCM
    header.write(struct.pack("<H", 1))  # AudioFormat = PCM
    header.write(struct.pack("<H", channels))
    header.write(struct.pack("<I", sample_rate))
    header.write(struct.pack("<I", byte_rate))
    header.write(struct.pack("<H", block_align))
    header.write(struct.pack("<H", sample_width * 8))  # BitsPerSample
    # data subchunk
    header.write(b"data")
    header.write(struct.pack("<I", subchunk2_size))
    return header.getvalue()


def _parse_mp3_frame(
    buffer: memoryview,
) -> Tuple[Optional[bytes], Optional[int], Optional[float], Optional[int]]:
    """
    Parse the first complete MP3 frame from the buffer.

    Args:
        buffer (memoryview): Buffer containing MP3 byte data.

    Returns:
        Tuple:
            frame_bytes (bytes | None): Full frame bytes, empty bytes if invalid, None if incomplete.
            frame_length (int | None): Length in bytes, None if invalid or incomplete.
            frame_duration_ms (float | None): Duration in ms, None if invalid or incomplete.
            sample_rate (int | None): Sample rate in Hz, None if invalid or incomplete.
    """
    sync_offset: int = -1
    for i in range(len(buffer) - 1):
        if buffer[i] == 0xFF and (buffer[i + 1] & 0xE0) == 0xE0:
            sync_offset = i
            break
    if sync_offset == -1:
        return None, None, None, None

    if sync_offset + 4 > len(buffer):
        return None, None, None, None

    header_bytes = buffer[sync_offset : sync_offset + 4]
    try:
        h1, h2, h3, h4 = struct.unpack("!BBBB", header_bytes)
        # Extract bits
        mpeg_version_bits = (h2 >> 3) & 0b11
        layer_bits = (h2 >> 1) & 0b11
        bitrate_index = (h3 >> 4) & 0b1111
        samplerate_index = (h3 >> 2) & 0b11
        padding_bit = (h3 >> 1) & 0b1

        # Validate
        if (
            mpeg_version_bits == 1
            or layer_bits == 0
            or bitrate_index in (0, 15)
            or samplerate_index == 3
        ):
            return b"", None, None, None

        mpeg_map = {0: 2.5, 2: 2, 3: 1}
        layer_map = {1: 3, 2: 2, 3: 1}
        mpeg_version = mpeg_map[mpeg_version_bits]
        layer = layer_map[layer_bits]
        sample_rate_hz = SAMPLE_RATES[mpeg_version][samplerate_index]
        bitrate_kbps = BITRATES[(mpeg_version if mpeg_version == 1 else 2, layer)][
            bitrate_index
        ]

        samples = SAMPLES_PER_FRAME[(mpeg_version, layer)]
        padding_size = 1 if padding_bit else 0
        frame_length = (
            samples // 8 * bitrate_kbps * 1000
        ) // sample_rate_hz + padding_size
        if frame_length <= 0 or sync_offset + frame_length > len(buffer):
            return None, None, None, None
        frame_duration_ms = (samples / sample_rate_hz) * 1000
        frame_bytes = bytes(buffer[sync_offset : sync_offset + frame_length])
        return frame_bytes, frame_length, frame_duration_ms, sample_rate_hz

    except Exception as e:
        logger.warning(f"MP3 parse error at offset {sync_offset}: {e}")
        return b"", None, None, None


def _robust_get_volume_by_chunks(
    audio: AudioSegment, chunk_length_ms: int
) -> List[float]:
    """
    Compute normalized RMS volume for fixed-size chunks of audio.

    Args:
        audio (AudioSegment): Loaded audio segment.
        chunk_length_ms (int): Chunk duration in milliseconds.

    Returns:
        List[float]: Normalized RMS volume per chunk (0.0-1.0).
    """
    if len(audio) == 0:
        return []
    chunks = make_chunks(audio, chunk_length_ms)
    if not chunks:
        return []

    rms_values: List[int] = [chunk.rms for chunk in chunks]
    max_rms = max(rms_values, default=0)
    if max_rms == 0:
        return [0.0] * len(rms_values)
    return [(rms / max_rms) for rms in rms_values]


async def stream_audio_payload(
    audio_stream_iterator: AsyncIterator[bytes],
    audio_format: str,
    sentence_index: int,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    sample_width: Optional[int] = None,
    buffer_duration_ms: int = 1000,
    chunk_length_ms: int = 20,
    display_text: Optional[DisplayText] = None,
    actions: Optional[Actions] = None,
    forwarded: bool = False,
) -> AsyncGenerator[Dict[str, Any]]:
    """
    Process an async iterator of MP3/WAV chunks, buffer into segments, compute volumes,
    and yield WAV-based payloads ready for WebSocket streaming.

    Args:
        audio_stream_iterator (AsyncIterator[bytes]): Source of raw audio bytes.
        audio_format (str): 'mp3' or 'wav'.
        sentence_index (int): Index of the sentence being processed.
        sample_rate (Optional[int]): Required for WAV.
        channels (Optional[int]): Required for WAV.
        sample_width (Optional[int]): Bytes per sample for WAV.
        buffer_duration_ms (int): Buffer duration in ms for each segment.
        chunk_length_ms (int): Duration in ms for volume analysis. It's 20ms.
        display_text (Optional[DisplayText]): Static display text. It's only included in the first payload of the sentence.
        actions (Optional[Actions]): Actions associated with the audio. It's only included in the first payload of the sentence.
        forwarded (bool): Static forwarded flag. It's only included in the first payload of the sentence.

    Yields:
        Dict[str, Any]: Payload dict containing audio, volumes, metadata.
    """
    if audio_format not in ("mp3", "wav"):
        logger.error(f"TTS stream: Unsupported audio format: {audio_format}")
        yield {"type": "error", "message": f"Unsupported format: {audio_format}"}
        return
    if audio_format == "wav" and not all([sample_rate, channels, sample_width]):
        logger.error(
            "TTS stream: Missing WAV parameters: sample_rate, channels, sample_width"
        )
        yield {"type": "error", "message": "Missing WAV parameters"}
        return

    input_buffer = bytearray()
    mp3_frame_buffer: Deque[Tuple[bytes, float]] = deque()
    buffered_duration_ms = 0.0
    first_mp3_rate: Optional[int] = None
    sub_sentence_index = 0

    async def _process_and_yield_segment(segment_bytes: bytes, format_for_pydub: str):
        nonlocal sub_sentence_index
        try:
            audio_seg = AudioSegment.from_file(
                io.BytesIO(segment_bytes), format=format_for_pydub
            )
            wav_bytes = audio_seg.export(format="wav").read()
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            volumes = _robust_get_volume_by_chunks(audio_seg, chunk_length_ms)

            payload: Dict[str, Any] = {
                "type": "audio",
                "audio": audio_b64,
                "volumes": volumes,
                "slice_length": chunk_length_ms,
                "sentence_index": sentence_index,
                "sub_sentence_index": sub_sentence_index,
                "end_of_sentence": False,
                "display_text": None,
                "actions": None,
                "forwarded": None,
                "segment_duration_ms": len(audio_seg),  # Add the segment duration here
            }

            if sub_sentence_index == 0:
                payload["display_text"] = (
                    display_text.to_dict() if display_text else None
                )
                payload["actions"] = actions.to_dict() if actions else None
                payload["forwarded"] = forwarded

            yield {
                k: v for k, v in payload.items() if v is not None or k in ["forwarded"]
            }
            sub_sentence_index += 1
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Processing error: {e}",
                "sentence_index": sentence_index,
            }

    try:
        async for chunk in audio_stream_iterator:
            # Skip non-bytes
            if not isinstance(chunk, (bytes, bytearray)):
                logger.warning(f"Skipping non-bytes chunk: {type(chunk)}")
                continue
            input_buffer.extend(chunk)

            while True:
                segment_bytes: Optional[bytes] = None
                format_for_pydub: str

                if audio_format == "mp3":
                    buffer_view = memoryview(bytes(input_buffer))
                    consumed = 0
                    while True:
                        frame, length, dur_ms, rate = _parse_mp3_frame(
                            buffer_view[consumed:]
                        )
                        if frame is None:
                            break
                        if frame == b"":
                            consumed += 1
                            continue
                        if length is None or dur_ms is None or rate is None:
                            consumed += 1
                            continue

                        if first_mp3_rate is None:
                            first_mp3_rate = rate
                        mp3_frame_buffer.append((frame, dur_ms))
                        buffered_duration_ms += dur_ms
                        consumed += length

                    if consumed:
                        del input_buffer[:consumed]
                    if buffered_duration_ms >= buffer_duration_ms and mp3_frame_buffer:
                        frames, seg_dur = [], 0.0
                        while mp3_frame_buffer and seg_dur < buffer_duration_ms:
                            f, dm = mp3_frame_buffer.popleft()
                            frames.append(f)
                            seg_dur += dm
                        buffered_duration_ms -= seg_dur
                        segment_bytes = b"".join(frames)
                        format_for_pydub = "mp3"
                    else:
                        break

                else:  # WAV handling
                    assert sample_rate and channels and sample_width
                    bytes_per_ms = sample_rate * channels * sample_width / 1000.0
                    buf_dur = len(input_buffer) / bytes_per_ms
                    if buf_dur < buffer_duration_ms:
                        break
                    to_proc = int(buffer_duration_ms * bytes_per_ms)
                    block = channels * sample_width
                    to_proc = (to_proc // block) * block
                    data = bytes(input_buffer[:to_proc])
                    del input_buffer[:to_proc]
                    header = _create_wav_header(
                        sample_rate, channels, sample_width, to_proc // block
                    )
                    segment_bytes = header + data
                    format_for_pydub = "wav"

                if segment_bytes is not None:
                    async for payload in _process_and_yield_segment(
                        segment_bytes, format_for_pydub
                    ):
                        yield payload
                    continue

                break

        # Handle remaining buffer at end of stream
        while True:
            segment_bytes: Optional[bytes] = None
            format_for_pydub: str

            if audio_format == "mp3":
                buffer_view = memoryview(bytes(input_buffer))
                consumed = 0
                while True:
                    frame, length, dur_ms, rate = _parse_mp3_frame(
                        buffer_view[consumed:]
                    )
                    if frame is None:
                        break
                    if frame == b"":
                        consumed += 1
                        continue
                    if length is None or dur_ms is None or rate is None:
                        consumed += 1
                        continue

                    if first_mp3_rate is None:
                        first_mp3_rate = rate
                    mp3_frame_buffer.append((frame, dur_ms))
                    buffered_duration_ms += dur_ms
                    consumed += length

                if consumed:
                    del input_buffer[:consumed]
                if buffered_duration_ms >= buffer_duration_ms and mp3_frame_buffer:
                    frames, seg_dur = [], 0.0
                    while mp3_frame_buffer and seg_dur < buffer_duration_ms:
                        f, dm = mp3_frame_buffer.popleft()
                        frames.append(f)
                        seg_dur += dm
                    buffered_duration_ms -= seg_dur
                    segment_bytes = b"".join(frames)
                    format_for_pydub = "mp3"
                else:
                    break

            else:  # WAV handling
                assert sample_rate and channels and sample_width
                bytes_per_ms = sample_rate * channels * sample_width / 1000.0
                buf_dur = len(input_buffer) / bytes_per_ms
                if buf_dur < buffer_duration_ms:
                    break
                to_proc = int(buffer_duration_ms * bytes_per_ms)
                block = channels * sample_width
                to_proc = (to_proc // block) * block
                data = bytes(input_buffer[:to_proc])
                del input_buffer[:to_proc]
                header = _create_wav_header(
                    sample_rate, channels, sample_width, to_proc // block
                )
                segment_bytes = header + data
                format_for_pydub = "wav"

            if segment_bytes is not None:
                async for payload in _process_and_yield_segment(
                    segment_bytes, format_for_pydub
                ):
                    yield payload
                continue

            break

        # Final end-of-sentence signal
        yield {
            "type": "audio",
            "audio": None,
            "volumes": [],
            "slice_length": chunk_length_ms,
            "sentence_index": sentence_index,
            "sub_sentence_index": sub_sentence_index,
            "end_of_sentence": True,
        }

    except asyncio.CancelledError:
        yield {"type": "info", "message": "Stream cancelled"}
    except Exception as e:
        yield {"type": "error", "message": f"Fatal error: {e}"}
    finally:
        logger.info("Audio stream processing finished.")


async def dummy_audio_iterator(
    file_path: str, chunk_size: int = 4096
) -> AsyncGenerator[bytes, None]:
    """
    Simulate async audio chunks by reading a local file.

    Args:
        file_path (str): Path to audio file.
        chunk_size (int): Byte size per chunk.

    Yields:
        bytes: Next audio chunk.
    """
    try:
        with open(file_path, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                yield data
                await asyncio.sleep(0.01)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return


def save_payload_to_json(
    payload: Dict[str, Any], directory: str = "./payloads", prefix: str = "payload"
) -> str:
    """
    Save a payload dictionary to a JSON file with a timestamped name.

    Args:
        payload (Dict[str, Any]): Payload to persist.
        directory (str): Directory to save the JSON file.
        prefix (str): Filename prefix.

    Returns:
        str: Path to the saved JSON file.
    """
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.json"
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved payload: {filepath}")
    return filepath


async def main() -> None:
    """
    Example entrypoint: streams a test MP3/WAV file, saves payloads.
    """
    MP3_FILE = "./test.mp3"
    selected_format = "mp3"

    if selected_format == "mp3":
        iterator = dummy_audio_iterator(MP3_FILE)
        stream_gen = stream_audio_payload(
            iterator,
            audio_format="mp3",
            sentence_index=0,
            buffer_duration_ms=1500,
            display_text=DisplayText(text="Example text"),
            actions=Actions(["example_action"]),  # Fixed: use positional argument
            forwarded=True,
        )
    else:
        print("Invalid format")
        return

    payload_count = 0
    async for payload in stream_gen:
        payload_count += 1
        save_payload_to_json(payload, prefix=f"{selected_format}_{payload_count:03d}")
        if payload.get("type") == "audio":
            print(f"Received audio segment #{payload_count}")
        elif payload.get("type") == "error":
            print(f"Error: {payload['message']}")


if __name__ == "__main__":
    asyncio.run(main())
