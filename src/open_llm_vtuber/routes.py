from uuid import uuid4
from fastapi import APIRouter, WebSocket, UploadFile, File, Form, HTTPException
from starlette.websockets import WebSocketDisconnect
from loguru import logger
from typing import Optional
import numpy as np
import tempfile
import os
from langdetect import detect
from datetime import datetime
from fastapi.responses import JSONResponse, Response
import json
from pydub import AudioSegment

from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler
from .proxy_handler import ProxyHandler


def init_client_ws_route(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling the `/client-ws` WebSocket connections.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()
    ws_handler = WebSocketHandler(default_context_cache)

    @router.websocket("/client-ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for client connections"""
        await websocket.accept()
        client_uid = str(uuid4())

        try:
            await ws_handler.handle_new_connection(websocket, client_uid)
            await ws_handler.handle_websocket_communication(websocket, client_uid)
        except WebSocketDisconnect:
            await ws_handler.handle_disconnect(client_uid)
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            await ws_handler.handle_disconnect(client_uid)
            raise

    return router


def init_proxy_route(server_url: str) -> APIRouter:
    """
    Create and return API routes for handling proxy connections.

    Args:
        server_url: The WebSocket URL of the actual server

    Returns:
        APIRouter: Configured router with proxy WebSocket endpoint
    """
    router = APIRouter()
    proxy_handler = ProxyHandler(server_url)

    @router.websocket("/proxy-ws")
    async def proxy_endpoint(websocket: WebSocket):
        """WebSocket endpoint for proxy connections"""
        try:
            await proxy_handler.handle_client_connection(websocket)
        except Exception as e:
            logger.error(f"Error in proxy connection: {e}")
            raise

    return router


def init_webtool_routes(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling web tool interactions.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()

    @router.get("/web-tool")
    async def web_tool_redirect():
        """Redirect /web-tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/web_tool")
    async def web_tool_redirect_alt():
        """Redirect /web_tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/live2d-models/info")
    async def get_live2d_folder_info():
        """Get information about available Live2D models"""
        live2d_dir = "live2d-models"
        if not os.path.exists(live2d_dir):
            return JSONResponse(
                {"error": "Live2D models directory not found"}, status_code=404
            )

        valid_characters = []
        supported_extensions = [".png", ".jpg", ".jpeg"]

        for entry in os.scandir(live2d_dir):
            if entry.is_dir():
                folder_name = entry.name.replace("\\", "/")
                model3_file = os.path.join(
                    live2d_dir, folder_name, f"{folder_name}.model3.json"
                ).replace("\\", "/")

                if os.path.isfile(model3_file):
                    # Find avatar file if it exists
                    avatar_file = None
                    for ext in supported_extensions:
                        avatar_path = os.path.join(
                            live2d_dir, folder_name, f"{folder_name}{ext}"
                        )
                        if os.path.isfile(avatar_path):
                            avatar_file = avatar_path.replace("\\", "/")
                            break

                    valid_characters.append(
                        {
                            "name": folder_name,
                            "avatar": avatar_file,
                            "model_path": model3_file,
                        }
                    )
        return JSONResponse(
            {
                "type": "live2d-models/info",
                "count": len(valid_characters),
                "characters": valid_characters,
            }
        )

    @router.post("/asr")
    async def transcribe_audio(file: UploadFile = File(...)):
        """
        Endpoint for transcribing audio using the ASR engine
        """
        logger.info(f"Received audio file for transcription: {file.filename}")

        try:
            contents = await file.read()

            # Validate minimum file size
            if len(contents) < 44:  # Minimum WAV header size
                raise ValueError("Invalid WAV file: File too small")

            # Decode the WAV header and get actual audio data
            wav_header_size = 44  # Standard WAV header size
            audio_data = contents[wav_header_size:]

            # Validate audio data size
            if len(audio_data) % 2 != 0:
                raise ValueError("Invalid audio data: Buffer size must be even")

            # Convert to 16-bit PCM samples to float32
            try:
                audio_array = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            except ValueError as e:
                raise ValueError(
                    f"Audio format error: {str(e)}. Please ensure the file is 16-bit PCM WAV format."
                )

            # Validate audio data
            if len(audio_array) == 0:
                raise ValueError("Empty audio data")

            text = await default_context_cache.asr_engine.async_transcribe_np(
                audio_array
            )
            logger.info(f"Transcription result: {text}")
            return {"text": text}

        except ValueError as e:
            logger.error(f"Audio format error: {e}")
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=400,
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return Response(
                content=json.dumps(
                    {"error": "Internal server error during transcription"}
                ),
                status_code=500,
                media_type="application/json",
            )

    @router.websocket("/tts-ws")
    async def tts_endpoint(websocket: WebSocket):
        """WebSocket endpoint for TTS generation"""
        await websocket.accept()
        logger.info("TTS WebSocket connection established")

        try:
            while True:
                data = await websocket.receive_json()
                text = data.get("text")
                if not text:
                    continue

                logger.info(f"Received text for TTS: {text}")

                # Split text into sentences
                sentences = [s.strip() for s in text.split(".") if s.strip()]

                try:
                    # Generate and send audio for each sentence
                    for sentence in sentences:
                        sentence = sentence + "."  # Add back the period
                        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
                        audio_path = (
                            await default_context_cache.tts_engine.async_generate_audio(
                                text=sentence, file_name_no_ext=file_name
                            )
                        )
                        logger.info(
                            f"Generated audio for sentence: {sentence} at: {audio_path}"
                        )

                        await websocket.send_json(
                            {
                                "status": "partial",
                                "audioPath": audio_path,
                                "text": sentence,
                            }
                        )

                    # Send completion signal
                    await websocket.send_json({"status": "complete"})

                except Exception as e:
                    logger.error(f"Error generating TTS: {e}")
                    await websocket.send_json({"status": "error", "message": str(e)})

        except WebSocketDisconnect:
            logger.info("TTS WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in TTS WebSocket connection: {e}")
            await websocket.close()

    @router.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible endpoint)"""
        return {
            "object": "list",
            "data": [
                {
                    "id": "indextts2",
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "custom",
                },
                {
                    "id": "funASR",
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "custom",
                }
            ]
        }

    @router.post("/v1/audio/transcriptions")
    async def stt_endpoint(
        file: UploadFile = File(...),
        model: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(0.0),
    ):
        """
        Transcribe audio file using the ASR engine.
        Compatible with OpenAI's /v1/audio/transcriptions API.
        
        Args:
            file: Audio file to transcribe (flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm)
            model: Model to use (ignored, uses configured ASR engine)
            language: Language of the audio (optional)
            prompt: Optional text to guide the model's style
            response_format: Format of the response (json, text, srt, verbose_json, vtt)
            temperature: Sampling temperature (0-1)
        
        Returns:
            Transcription result in the specified format
        """
        try:
            # Read the uploaded file
            audio_data = await file.read()
            
            # Get file extension
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            # Convert audio to numpy array
            audio_np = await _convert_audio_to_numpy(audio_data, file_extension)
            
            # Transcribe using the ASR engine
            transcription = await default_context_cache.asr_engine.async_transcribe_np(audio_np)
            
            detected_language = None
            try:
                if transcription:
                    detected_language = detect(transcription)
            except Exception:
                detected_language = None
            language_out = detected_language or "unknown"
            
            # Format response based on response_format
            if response_format == "text":
                return transcription
            elif response_format == "json":
                return {"text": transcription}
            elif response_format == "verbose_json":
                return {
                    "task": "transcribe",
                    "language": language_out,
                    "duration": len(audio_np) / default_context_cache.asr_engine.SAMPLE_RATE,
                    "text": transcription,
                }
            elif response_format in ["srt", "vtt"]:
                # Simple implementation - return as single segment
                if response_format == "srt":
                    return f"1\n00:00:00,000 --> 00:00:10,000\n{transcription}\n"
                else:  # vtt
                    return f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{transcription}\n"
            else:
                return {"text": transcription}
                
        except Exception as e:
            logger.error(f"Error in transcription endpoint: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    return router


async def _convert_audio_to_numpy(audio_data: bytes, file_extension: str) -> np.ndarray:
    """
    Convert audio file bytes to numpy array suitable for ASR processing.
    
    Args:
        audio_data: Raw audio file bytes
        file_extension: File extension (e.g., '.wav', '.mp3', '.ogg')
    
    Returns:
        numpy array of audio samples normalized to [-1, 1]
    """
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Load audio using pydub
            audio = AudioSegment.from_file(temp_file_path)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Convert to 16kHz sample rate (standard for ASR)
            audio = audio.set_frame_rate(16000)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # Normalize to [-1, 1]
            if audio.sample_width == 2:  # 16-bit
                samples = samples.astype(np.float32) / 32768.0
            elif audio.sample_width == 4:  # 32-bit
                samples = samples.astype(np.float32) / 2147483648.0
            else:
                samples = samples.astype(np.float32) / 128.0
            
            return samples
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error converting audio to numpy: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process audio file: {str(e)}")
