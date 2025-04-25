"""Prepares audio payload for sending to the client.
This module only handles non-streaming tts audio payloads.
See `audio_stream_payload.py` for streaming tts audio payloads.
"""

import base64
from pydub import AudioSegment
from pydub.utils import make_chunks
from ..agent.output_types import Actions
from ..agent.output_types import DisplayText


def _get_volume_by_chunks(audio: AudioSegment, chunk_length_ms: int) -> list:
    """
    Calculate the normalized volume (RMS) for each chunk of the audio.

    Parameters:
        audio (AudioSegment): The audio segment to process.
        chunk_length_ms (int): The length of each audio chunk in milliseconds.

    Returns:
        list: Normalized volumes for each chunk.
    """
    chunks = make_chunks(audio, chunk_length_ms)
    volumes = [chunk.rms for chunk in chunks]
    max_volume = max(volumes)
    if max_volume == 0:
        raise ValueError("Audio is empty or all zero.")
    return [volume / max_volume for volume in volumes]


def prepare_audio_payload(
    sentence_index: int,
    audio_path: str | None,
    response_uid: str | None = None,
    chunk_length_ms: int = 20,
    display_text: DisplayText = None,
    actions: Actions = None,
    forwarded: bool = False,
    cleanup: bool = True,
) -> dict[str, any]:
    """
    Prepares the audio payload for sending to the client.
    If audio_path is None, returns a payload with audio=None for silent display.

    Parameters:
        sentence_index (int): The index of this audio (sentence) in the AI response
        audio_path (str | None): The path to the audio file to be processed, or None for silent display
        response_uid (str | None): Unique identifier for the response. Defaults to None.
            if None, the frontend should treat this payload as a standalone message.
        chunk_length_ms (int): The length of each audio chunk in milliseconds
        display_text (DisplayText, optional): Text to be displayed with the audio
        actions (Actions, optional): Actions associated with the audio
        forwarded (bool): idk. If you have it, pass it.
        cleanup (bool): Whether to clean up the audio file after processing
    Returns:
        dict: The audio payload to be sent
    """
    if isinstance(display_text, DisplayText):
        display_text = display_text.to_dict()

    if not audio_path:
        # Return payload for silent display
        return {
            "type": "audio",
            "response_id": response_uid,
            "audio": None,
            "sentence_index": sentence_index,
            "sub_sentence_index": 0,
            "end_of_sentence": True,
            "volumes": [],
            "slice_length": chunk_length_ms,
            "display_text": display_text,
            "actions": actions.to_dict() if actions else None,
            "forwarded": forwarded,
        }

    try:
        audio = AudioSegment.from_file(audio_path)
        audio_bytes = audio.export(format="wav").read()
    except Exception as e:
        raise ValueError(
            f"Error loading or converting generated audio file to wav file '{audio_path}': {e}"
        )
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    volumes = _get_volume_by_chunks(audio, chunk_length_ms)

    payload = {
        "type": "audio",
        "response_id": response_uid,
        "audio": audio_base64,
        "sentence_index": sentence_index,
        "sub_sentence_index": 0,
        "end_of_sentence": True,
        "volumes": volumes,
        "slice_length": chunk_length_ms,
        "display_text": display_text,
        "actions": actions.to_dict() if actions else None,
        "forwarded": forwarded,
    }

    return payload
