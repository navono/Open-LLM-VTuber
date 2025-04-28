"""
Module for streaming audio payload processing.

This module provides functionality to process asynchronous streams of audio chunks
(MP3 or WAV) and yield json payloads ready to be sent to the client.

Look at the `stream_audio_payload` function for the core functionality.
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


def _detect_audio_format_and_wav_params(
    buffer: bytes,
) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    """
    Detect audio format ('mp3' or 'wav') from initial buffer bytes and
    parse WAV parameters if it's a WAV file.

    Args:
        buffer (bytes): Initial bytes from the audio stream

    Returns:
        Tuple:
            format (str): 'mp3' or 'wav'
            sample_rate (int | None): samples per second (Hz) for WAV, None for MP3
            channels (int | None): number of audio channels for WAV, None for MP3
            sample_width (int | None): bytes per sample for WAV, None for MP3
    """
    # First check for WAV format
    if len(buffer) >= 44 and buffer[:4] == b"RIFF" and buffer[8:12] == b"WAVE":
        # Locate 'fmt ' subchunk
        fmt_offset = buffer.find(b"fmt ")
        if fmt_offset != -1 and len(buffer) >= fmt_offset + 24:
            try:
                sample_rate = struct.unpack(
                    "<I", buffer[fmt_offset + 12 : fmt_offset + 16]
                )[0]
                channels = struct.unpack(
                    "<H", buffer[fmt_offset + 10 : fmt_offset + 12]
                )[0]
                bits_per_sample = struct.unpack(
                    "<H", buffer[fmt_offset + 22 : fmt_offset + 24]
                )[0]
                sample_width = bits_per_sample // 8
                logger.info(
                    f"Detected WAV format: {sample_rate}Hz, {channels} channels, {bits_per_sample}-bit"
                )
                return "wav", sample_rate, channels, sample_width
            except struct.error:
                logger.warning("Error parsing WAV header, falling back to MP3")

    # Check for MP3 format (look for sync word: 11 consecutive 1s)
    # We're looking for 0xFF followed by at least 0xE0 (top 3 bits are 1)
    for i in range(len(buffer) - 1):
        if buffer[i] == 0xFF and (buffer[i + 1] & 0xE0) == 0xE0:
            # Look further to see if we can find a valid MP3 frame
            frame, _, _, _ = _parse_mp3_frame(memoryview(buffer[i:]))
            if frame not in (None, b""):
                logger.info("Detected MP3 format")
                return "mp3", None, None, None

    # If we can't be sure, default to MP3
    logger.warning("Unable to detect format definitively, defaulting to MP3")
    return "mp3", None, None, None


async def stream_audio_payload(
    audio_stream_iterator: AsyncIterator[bytes],
    response_uid: str,
    audio_format: Optional[str] = None,
    sentence_index: int = 0,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    sample_width: Optional[int] = None,
    buffer_duration_ms: int = 2000,
    chunk_length_ms: int = 20,
    display_text: Optional[DisplayText] = None,
    actions: Optional[Actions] = None,
    forwarded: bool = False,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Process an async iterator of MP3/WAV chunks, buffer into segments, compute volumes,
    and yield WAV-based payloads ready for WebSocket streaming.

    Handles short audio streams correctly by yielding any remaining buffered audio
    at the end, even if it's shorter than buffer_duration_ms.

    Args:
        audio_stream_iterator (AsyncIterator[bytes]): Source of raw audio bytes.
        response_uid (str): Unique identifier for the response.
        audio_format (Optional[str]): 'mp3', 'wav', or None for auto-detection.
        sentence_index (int): Index of the sentence being processed.
        sample_rate (Optional[int]): Required for WAV if not auto-detected.
        channels (Optional[int]): Required for WAV if not auto-detected.
        sample_width (Optional[int]): Bytes per sample for WAV if not auto-detected.
        buffer_duration_ms (int): Buffer duration in ms for each segment.
        chunk_length_ms (int): Duration in ms for volume analysis. It's 20ms.
        display_text (Optional[DisplayText]): Static display text. It's only included in the first payload of the sentence.
        actions (Optional[Actions]): Actions associated with the audio. It's only included in the first payload of the sentence.
        forwarded (bool): Static forwarded flag. It's only included in the first payload of the sentence.

    Yields:
        Dict[str, Any]: Payload dict containing audio, volumes, metadata.
    """
    print(f"Sentence Index: {sentence_index}")
    initial_bytes_collected = bytearray()
    initial_chunks_for_rebuffering = []

    # --- Format Detection ---
    if audio_format is None or (
        audio_format == "wav" and not all([sample_rate, channels, sample_width])
    ):
        while len(initial_bytes_collected) < 128:  # Collect enough for detection
            try:
                # Use anext() to get the next item from the async iterator
                chunk = await asyncio.wait_for(
                    anext(audio_stream_iterator), timeout=5.0
                )  # Add timeout
                if not isinstance(chunk, (bytes, bytearray)):
                    logger.warning(
                        f"Skipping non-bytes chunk during detection: {type(chunk)}"
                    )
                    continue
                initial_chunks_for_rebuffering.append(chunk)  # Store for later use
                initial_bytes_collected.extend(chunk)
            except StopAsyncIteration:
                logger.info("Stream ended during initial format detection.")
                break  # Exit collection loop if stream ends early
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for initial chunk during format detection."
                )
                break
            except Exception as e:
                logger.error(f"Error reading initial chunk: {e}")
                break

        if not initial_bytes_collected:
            logger.error(
                "Stream is empty or initial read failed, cannot detect format or process."
            )
            yield {"type": "error", "message": "Empty or unreadable audio stream"}
            # Yield final marker immediately if stream was truly empty
            yield {
                "type": "audio",
                "response_id": response_uid,
                "audio": None,
                "volumes": [],
                "slice_length": chunk_length_ms,
                "sentence_index": sentence_index,
                "sub_sentence_index": 0,
                "end_of_sentence": True,
                "note": display_text.text if display_text else None,
            }
            return

        detected_format, detected_sr, detected_ch, detected_sw = (
            _detect_audio_format_and_wav_params(bytes(initial_bytes_collected))
        )
        # Use detected values only if not explicitly provided or if format is None
        if audio_format is None:
            audio_format = detected_format
        if audio_format == "wav":
            sample_rate = sample_rate or detected_sr
            channels = channels or detected_ch
            sample_width = sample_width or detected_sw

    # --- Validation ---
    if audio_format not in ("mp3", "wav"):
        logger.error(f"TTS stream: Unsupported audio format: {audio_format}")
        yield {"type": "error", "message": f"Unsupported format: {audio_format}"}
        return
    if audio_format == "wav" and not all([sample_rate, channels, sample_width]):
        logger.error(
            f"TTS stream: WAV parameters missing/undetected: SR={sample_rate}, CH={channels}, SW={sample_width}"
        )
        yield {"type": "error", "message": "Missing required WAV parameters"}
        return

    # --- Initialization ---
    input_buffer = bytearray()
    # Add back the bytes collected during detection
    if initial_chunks_for_rebuffering:
        for chunk in initial_chunks_for_rebuffering:
            input_buffer.extend(chunk)

    mp3_frame_buffer: Deque[Tuple[bytes, float]] = deque()
    buffered_duration_ms = 0.0
    first_mp3_rate: Optional[int] = None  # Still useful if MP3 stream is very short
    sub_sentence_counter = 0
    block_align = 0  # Initialize block_align
    if audio_format == "wav":
        block_align = channels * sample_width  # Calculate once

    # --- Payload Generation Helper ---
    async def _process_and_yield_segment(
        sub_sent_idx: int, segment_bytes: bytes, format_for_pydub: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            # Ensure non-empty bytes before processing
            if not segment_bytes:
                logger.warning(
                    f"Attempted to process empty segment for sub_sentence {sub_sent_idx}"
                )
                return

            audio_seg = AudioSegment.from_file(
                io.BytesIO(segment_bytes), format=format_for_pydub
            )
            # Skip empty segments potentially created by pydub from malformed data
            if len(audio_seg) == 0:
                logger.warning(
                    f"Created empty AudioSegment for sub_sentence {sub_sent_idx}, possibly due to invalid input bytes."
                )
                return

            wav_bytes = audio_seg.export(format="wav").read()
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            volumes = _robust_get_volume_by_chunks(audio_seg, chunk_length_ms)

            payload: Dict[str, Any] = {
                "type": "audio",
                "response_id": response_uid,
                "audio": audio_b64,
                "volumes": volumes,
                "slice_length": chunk_length_ms,
                "sentence_index": sentence_index,
                "sub_sentence_index": sub_sent_idx,
                "end_of_sentence": False,
                "display_text": None,
                "actions": None,
                "forwarded": None,  # Ensure None is default unless explicitly set
                "segment_duration_ms": len(audio_seg),
            }

            # Attach metadata only to the very first segment yielded
            if sub_sent_idx == 0:
                payload["display_text"] = (
                    display_text.to_dict() if display_text else None
                )
                payload["actions"] = actions.to_dict() if actions else None
                payload["forwarded"] = forwarded  # Use the boolean value directly

            # Filter out None values for display_text and actions before yielding
            yield {
                k: v
                for k, v in payload.items()
                if not (k in ["display_text", "actions", "forwarded"] and v is None)
            }

        except Exception as e:
            logger.error(
                f"Error processing audio segment for sub_sentence {sub_sent_idx}: {e}",
                exc_info=True,
            )
            yield {
                "type": "error",
                "message": f"Processing error: {e}",
                "sentence_index": sentence_index,
                "sub_sentence_index": sub_sent_idx,
                "note": display_text.text if display_text else None,
            }

    # --- Main Processing Loop ---
    try:
        async for chunk in audio_stream_iterator:
            if not isinstance(chunk, (bytes, bytearray)):
                logger.warning(f"Skipping non-bytes chunk: {type(chunk)}")
                continue

            if input_buffer:
                input_buffer = input_buffer + chunk
            else:
                input_buffer = bytearray(chunk)

            # Process buffer as long as enough data is available
            while True:
                segment_to_process: Optional[bytes] = None
                format_for_pydub: str = ""  # Initialize

                if audio_format == "mp3":
                    # Create a short-lived view; immediately drop after use
                    mv = memoryview(input_buffer)
                    consumed_bytes = 0
                    while True:  # Parse all available frames from current input_buffer
                        # Ensure we don't read past the buffer view
                        if consumed_bytes >= len(mv):
                            break
                        frame, length, dur_ms, rate = _parse_mp3_frame(
                            mv[consumed_bytes:]
                        )
                        if frame is None:
                            break  # Need more data
                        if (
                            frame == b""
                            or length is None
                            or dur_ms is None
                            or rate is None
                        ):
                            consumed_bytes += 1  # Skip invalid byte/header
                            continue

                        if first_mp3_rate is None:
                            first_mp3_rate = rate  # Capture rate from first valid frame
                        mp3_frame_buffer.append((frame, dur_ms))
                        buffered_duration_ms += dur_ms
                        consumed_bytes += length

                    if consumed_bytes > 0:
                        # Slice returns a new bytearray; old one still owned by mv
                        input_buffer = input_buffer[consumed_bytes:]

                    # Release the view
                    del mv

                    # Check if enough *duration* is buffered
                    if buffered_duration_ms >= buffer_duration_ms and mp3_frame_buffer:
                        frames_for_segment, segment_duration = [], 0.0
                        while (
                            mp3_frame_buffer and segment_duration < buffer_duration_ms
                        ):
                            f_bytes, f_dur = mp3_frame_buffer.popleft()
                            frames_for_segment.append(f_bytes)
                            segment_duration += f_dur
                        buffered_duration_ms -= (
                            segment_duration  # Adjust remaining buffered duration
                        )
                        segment_to_process = b"".join(frames_for_segment)
                        format_for_pydub = "mp3"
                    else:
                        break  # Not enough duration buffered yet, break inner while to get more chunks

                else:  # WAV handling
                    assert (
                        sample_rate and channels and sample_width
                    )  # Already validated
                    bytes_per_ms = sample_rate * channels * sample_width / 1000.0
                    # Prevent division by zero if params are somehow invalid (shouldn't happen)
                    if bytes_per_ms <= 0:
                        logger.error(
                            "Invalid WAV parameters lead to zero bytes_per_ms."
                        )
                        # Yield error and attempt to break cleanly
                        yield {
                            "type": "error",
                            "message": "Invalid WAV parameters",
                            "note": display_text.text if display_text else None,
                        }
                        break  # Break inner while

                    buffered_bytes_duration_ms = (
                        len(input_buffer) / bytes_per_ms if bytes_per_ms > 0 else 0
                    )

                    if buffered_bytes_duration_ms >= buffer_duration_ms:
                        bytes_to_process = int(buffer_duration_ms * bytes_per_ms)
                        # Align to block boundary
                        # block_align calculated during initialization
                        if block_align <= 0:  # Safety check
                            logger.error(
                                "Invalid WAV parameters lead to zero block_align."
                            )
                            yield {
                                "type": "error",
                                "message": "Invalid WAV parameters",
                                "note": display_text.text if display_text else None,
                            }
                            break  # Break inner while
                        # Ensure bytes_to_process is a multiple of block_align
                        bytes_to_process = (
                            bytes_to_process // block_align
                        ) * block_align

                        if bytes_to_process > 0 and bytes_to_process <= len(
                            input_buffer
                        ):
                            data_segment = bytes(input_buffer[:bytes_to_process])
                            input_buffer = input_buffer[
                                bytes_to_process:
                            ]  # Consume from buffer
                            num_frames = bytes_to_process // block_align
                            header = _create_wav_header(
                                sample_rate, channels, sample_width, num_frames
                            )
                            segment_to_process = header + data_segment
                            format_for_pydub = "wav"
                        elif bytes_to_process <= 0:
                            # Not enough bytes for even one block based on duration calculation
                            break  # Break inner while
                        else:  # bytes_to_process > len(input_buffer) - should not happen if buffered_bytes_duration_ms check is correct
                            logger.warning(
                                f"Calculated bytes_to_process ({bytes_to_process}) > available buffer ({len(input_buffer)}). Breaking inner loop."
                            )
                            break  # Break inner while
                    else:
                        break  # Not enough duration buffered yet, break inner while

                # --- Yield Segment if Ready ---
                if segment_to_process is not None:
                    processed_payload = False
                    async for payload in _process_and_yield_segment(
                        sub_sent_idx=sub_sentence_counter,
                        segment_bytes=segment_to_process,
                        format_for_pydub=format_for_pydub,
                    ):
                        yield payload
                        processed_payload = True  # Mark that at least one payload (even error) was yielded
                    # Increment only if the segment processing yielded something (wasn't skipped/empty)
                    if processed_payload:
                        sub_sentence_counter += 1
                    # Continue inner loop to process more data if available in buffer
                else:
                    # No segment was processed (either not enough data or MP3 parsing finished for this chunk)
                    break  # Break inner while loop to wait for more data from iterator

        # --- End of Stream Reached ---
        logger.info("Audio stream iterator finished. Processing remaining buffer.")

        # --- Process Final Remaining Data (The Fix) ---
        final_segment_bytes: Optional[bytes] = None
        final_format: str = ""

        if audio_format == "mp3":
            # Process any remaining *parsed* frames first
            if mp3_frame_buffer:
                logger.info(
                    f"Processing remaining {len(mp3_frame_buffer)} buffered MP3 frames ({buffered_duration_ms:.2f} ms)."
                )
                final_segment_bytes = b"".join([f for f, d in mp3_frame_buffer])
                final_format = "mp3"
                mp3_frame_buffer.clear()
                buffered_duration_ms = 0.0

            # Then, attempt to parse any remaining raw bytes in input_buffer
            if input_buffer:
                logger.info(
                    f"MP3 mode: Attempting to parse final {len(input_buffer)} bytes in input_buffer."
                )
                buffer_view = memoryview(input_buffer)
                consumed_bytes = 0
                final_frames_from_remains = []
                remaining_duration = 0.0
                while True:
                    if consumed_bytes >= len(buffer_view):
                        break
                    frame, length, dur_ms, rate = _parse_mp3_frame(
                        buffer_view[consumed_bytes:]
                    )
                    if frame is None:
                        break
                    if frame == b"" or length is None or dur_ms is None:
                        consumed_bytes += 1
                        continue
                    final_frames_from_remains.append(frame)
                    if dur_ms:
                        remaining_duration += dur_ms
                    consumed_bytes += length

                if final_frames_from_remains:
                    additional_bytes = b"".join(final_frames_from_remains)
                    if final_segment_bytes:  # Append if we already had buffered frames
                        final_segment_bytes += additional_bytes
                    else:  # Otherwise, this is the final segment
                        final_segment_bytes = additional_bytes
                        final_format = "mp3"
                    logger.info(
                        f"Parsed {len(final_frames_from_remains)} additional MP3 frames ({remaining_duration:.2f} ms) from remaining buffer."
                    )
                input_buffer = input_buffer[consumed_bytes:]  # Clear processed part
                if input_buffer:
                    logger.warning(
                        f"Discarding {len(input_buffer)} unparseable trailing bytes in MP3 stream."
                    )
                input_buffer.clear()

        else:  # WAV handling for remaining data
            assert sample_rate and channels and sample_width  # Should hold true
            if input_buffer:
                logger.info(
                    f"Processing remaining {len(input_buffer)} bytes of WAV data."
                )
                # Align to block boundary for the last chunk
                if block_align > 0:
                    bytes_to_process = (len(input_buffer) // block_align) * block_align
                    if bytes_to_process > 0:
                        data_segment = bytes(input_buffer[:bytes_to_process])
                        num_frames = bytes_to_process // block_align
                        header = _create_wav_header(
                            sample_rate, channels, sample_width, num_frames
                        )
                        final_segment_bytes = header + data_segment
                        final_format = "wav"
                        if len(input_buffer) > bytes_to_process:
                            logger.warning(
                                f"Discarding {len(input_buffer) - bytes_to_process} trailing WAV bytes (less than one block)."
                            )
                    else:
                        logger.warning(
                            f"Remaining {len(input_buffer)} WAV bytes are less than one block ({block_align} bytes), discarding."
                        )
                else:  # Should not happen due to earlier checks
                    logger.error(
                        "Block align is zero during final WAV processing, discarding remaining buffer."
                    )

                input_buffer.clear()  # Clear buffer after processing/discarding

        # Yield the final segment if any was collected
        if final_segment_bytes is not None and final_format:
            processed_final_payload = False
            async for payload in _process_and_yield_segment(
                sub_sent_idx=sub_sentence_counter,
                segment_bytes=final_segment_bytes,
                format_for_pydub=final_format,
            ):
                yield payload
                processed_final_payload = True
            # Increment after the final segment if it yielded something
            if processed_final_payload:
                sub_sentence_counter += 1

    except asyncio.CancelledError:
        logger.warning("Audio stream processing was cancelled.")
        yield {"type": "info", "message": "Stream cancelled"}
    except Exception as e:
        logger.error(f"Fatal error during audio stream processing: {e}", exc_info=True)
        yield {
            "type": "error",
            "message": f"Fatal processing error: {e}",
            "note": display_text.text if display_text else None,
        }
    finally:
        logger.info(
            f"Audio stream processing finished for sentence {sentence_index}. Yielding final marker."
        )
        # Yield the final end-of-sentence signal
        yield {
            "type": "audio",
            "response_id": response_uid,
            "audio": None,
            "volumes": [],
            "slice_length": chunk_length_ms,
            "sentence_index": sentence_index,
            # Use the final counter value for consistency
            "sub_sentence_index": sub_sentence_counter,
            "end_of_sentence": True,
            "note": display_text.text
            if display_text
            else None,  # Keep or remove 'note' as needed
        }


## ============ Testing code below ============ ##
# For testing purposes only


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
    # Configure Loguru basic console logging if not configured elsewhere
    # logger.add(lambda msg: print(msg, end=""), level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}") # Example config

    # Create dummy short MP3 file for testing the edge case
    try:
        short_mp3_path = "./short_test.mp3"
        # Ensure pydub is available for creating test files
        try:
            from pydub import AudioSegment

            silence = AudioSegment.silent(
                duration=1000
            )  # 1 second, shorter than default buffer
            silence.export(short_mp3_path, format="mp3")
            logger.info(f"Created short test file: {short_mp3_path}")
        except ImportError:
            logger.error(
                "pydub not found, cannot create test MP3 file. Skipping short MP3 test."
            )
            short_mp3_path = None  # Indicate file doesn't exist
        except Exception as e:
            logger.error(
                f"Failed to create short test MP3: {e}. Skipping short MP3 test."
            )
            short_mp3_path = None

        long_mp3_path = "./long_test.mp3"  # Assume this exists or create one
        if not os.path.exists(long_mp3_path):
            try:
                from pydub import AudioSegment

                AudioSegment.silent(duration=5000).export(long_mp3_path, format="mp3")
                logger.info(f"Created long test file: {long_mp3_path}")
            except ImportError:
                logger.error("pydub not found, cannot create long test MP3 file.")
            except Exception as e:
                logger.error(f"Failed to create long test MP3: {e}")
                # Consider exiting or skipping tests if creation fails

        test_cases = []
        if short_mp3_path and os.path.exists(short_mp3_path):
            test_cases.append(("short_mp3", short_mp3_path, "mp3", 1500))
        if os.path.exists(long_mp3_path):
            test_cases.append(("long_mp3", long_mp3_path, "mp3", 1500))
            # Add WAV tests if you have WAV files and ffmpeg/libs installed
            # ("short_wav", "short_test.wav", "wav", 1500),

        if not test_cases:
            logger.error("No valid test files found or created. Exiting test.")
            return

        # Dummy classes for testing if the real ones aren't available in this scope
        class DummyDisplayText:
            def __init__(self, text: str):
                self.text = text

            def to_dict(self):
                return {"text": self.text}

        class DummyActions:
            def __init__(self, actions: List[str]):
                self.actions = actions

            def to_dict(self):
                return {"actions": self.actions}

        for name, file_path, fmt, buf_dur in test_cases:
            logger.info(f"\n--- Running test: {name} ---")
            payload_count = 0
            iterator = dummy_audio_iterator(file_path)
            stream_gen = stream_audio_payload(
                iterator,
                response_uid=f"test_{name}_{datetime.now().isoformat()}",
                audio_format=fmt,
                sentence_index=0,
                buffer_duration_ms=buf_dur,
                # Provide dummy WAV params if testing WAV without auto-detection
                # sample_rate=44100, channels=1, sample_width=2,
                display_text=DummyDisplayText(text=f"Test: {name}"),
                actions=DummyActions([f"action_{name}"]),
                forwarded=(name == "long_mp3"),  # Example variation
            )

            async for payload in stream_gen:
                payload_count += 1
                save_payload_to_json(payload, prefix=f"{name}_{payload_count:03d}")
                p_type = payload.get("type")
                if p_type == "audio":
                    eos = payload.get("end_of_sentence")
                    audio_present = payload.get("audio") is not None
                    sub_idx = payload.get("sub_sentence_index")
                    duration = payload.get("segment_duration_ms", "N/A")
                    print(
                        f"Received Payload #{payload_count}: Type={p_type}, SubIdx={sub_idx}, AudioPresent={audio_present}, EOS={eos}, Duration={duration}"
                    )
                elif p_type == "error":
                    print(
                        f"Received Payload #{payload_count}: Type={p_type}, Message={payload.get('message')}"
                    )
                    logger.error(
                        f"Test {name} encountered error: {payload.get('message')}"
                    )
                else:
                    print(f"Received Payload #{payload_count}: Type={p_type}")

            logger.info(
                f"--- Test {name} finished. Total payloads: {payload_count} ---"
            )

        # Clean up dummy file
        if short_mp3_path and os.path.exists(short_mp3_path):
            try:
                os.remove(short_mp3_path)
                logger.info(f"Removed short test file: {short_mp3_path}")
            except OSError as e:
                logger.warning(f"Could not remove test file {short_mp3_path}: {e}")
        # if os.path.exists(long_mp3_path): os.remove(long_mp3_path) # Keep if needed elsewhere

    except Exception as e:
        logger.error(f"Error during test execution: {e}", exc_info=True)


if __name__ == "__main__":
    # Configure Loguru basic console logging
    # Note: Avoid adding multiple handlers if already configured elsewhere
    import sys

    logger.remove()  # Remove default handler to avoid duplicates if run multiple times
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    )
    asyncio.run(main())
