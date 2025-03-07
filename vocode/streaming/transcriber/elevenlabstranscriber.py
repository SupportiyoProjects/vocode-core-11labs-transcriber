import asyncio
import base64
import json
import queue
from datetime import datetime, timezone
from typing import List, Literal, Optional

import aiohttp
import sentry_sdk
from loguru import logger
from pydantic.v1 import Field

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import (
    ElevenLabsScribeTranscriberConfig,
    TranscriberConfig,
    Transcription,
)
from vocode.streaming.transcriber.base_transcriber import BaseThreadAsyncTranscriber
from vocode.utils.sentry_utils import CustomSentrySpans, sentry_create_span


class ElevenLabsScribeTranscriber(BaseThreadAsyncTranscriber[ElevenLabsScribeTranscriberConfig]):
    """
    Transcriber that uses ElevenLabs' Scribe speech-to-text service.
    
    This transcriber collects audio chunks, processes them in batches, and sends them
    to the ElevenLabs Scribe API for transcription.
    """
    
    ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/speech-to-text/convert"
    
    def __init__(
        self,
        transcriber_config: ElevenLabsScribeTranscriberConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(transcriber_config)
        
        self.api_key = api_key or getenv("ELEVEN_LABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key not provided. Set ELEVEN_LABS_API_KEY environment variable or pass it as a parameter.")
        
        self._ended = False
        self.is_ready = True
        self.buffer = bytearray()
        self.buffer_size_bytes = int(self.transcriber_config.sampling_rate * 
                                    self.transcriber_config.buffer_size_seconds * 
                                    (2 if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16 else 1))
        
        # Configure session for API requests
        self.session = None
    
    async def _create_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def _close_session(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def transcribe_audio(self, audio_data: bytes):
        """Send audio data to ElevenLabs Scribe API for transcription."""
        await self._create_session()
        
        # Prepare the audio data
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Determine content type based on audio encoding
        content_type = "audio/wav"
        if self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
            content_type = "audio/mulaw"
        elif self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
            content_type = "audio/l16"
        
        # Prepare the request payload
        payload = {
            "audio": audio_base64,
            "model_id": self.transcriber_config.model_id,
            "content_type": content_type,
            "sample_rate": self.transcriber_config.sampling_rate,
        }
        
        # Add optional parameters if specified
        if self.transcriber_config.language:
            payload["language"] = self.transcriber_config.language
        
        if self.transcriber_config.detect_language:
            payload["detect_language"] = True
            
        if self.transcriber_config.transcription_hints:
            payload["transcription_hints"] = self.transcriber_config.transcription_hints
            
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        
        try:
            async with self.session.post(
                self.ELEVENLABS_API_URL,
                headers=headers,
                json=payload,
                timeout=10
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ElevenLabs Scribe API error: {response.status}, {error_text}")
                    return None
                
                result = await response.json()
                return result
        except Exception as e:
            logger.error(f"Error calling ElevenLabs Scribe API: {str(e)}")
            return None
    
    def process_transcription_result(self, result):
        """Process the transcription result from ElevenLabs Scribe API."""
        if not result or "text" not in result:
            return
        
        # Create a span for latency tracking
        sentry_create_span(
            sentry_callable=sentry_sdk.start_span,
            op=CustomSentrySpans.LATENCY_OF_CONVERSATION,
            start_timestamp=datetime.now(tz=timezone.utc),
        )
        
        # Extract the transcribed text
        text = result["text"]
        
        # Get confidence if available
        confidence = result.get("confidence", 1.0)
        
        # Produce the transcription
        self.produce_nonblocking(
            Transcription(message=text, confidence=confidence, is_final=True)
        )
    
    def _run_loop(self):
        """Main processing loop that collects audio chunks and sends them for transcription."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while not self._ended:
            # Get audio chunks from the queue
            try:
                chunk = self.input_janus_queue.sync_q.get(timeout=5)
            except queue.Empty:
                continue
            
            if chunk is None:
                break
            
            # Add chunk to buffer
            self.buffer.extend(chunk)
            
            # If buffer is large enough, process it
            if len(self.buffer) >= self.buffer_size_bytes:
                # Copy buffer to avoid race conditions
                audio_data = bytes(self.buffer)
                self.buffer = bytearray()
                
                # Process the audio data
                result = loop.run_until_complete(self.transcribe_audio(audio_data))
                if result:
                    self.process_transcription_result(result)
        
        # Clean up
        loop.run_until_complete(self._close_session())
        loop.close()
    
    async def terminate(self):
        """Clean up resources when the transcriber is terminated."""
        self._ended = True
        await self._close_session()
        await super().terminate()


class ElevenLabsScribeTranscriberConfig(TranscriberConfig):
    """Configuration for the ElevenLabs Scribe transcriber."""
    
    transcriber_type: Literal["eleven_labs_scribe"] = Field("eleven_labs_scribe", alias="type")
    
    # ElevenLabs Scribe specific parameters
    model_id: str = "scribe_v1"  # Default model
    api_key: Optional[str] = None
    language: Optional[str] = None
    detect_language: bool = False
    transcription_hints: Optional[List[str]] = None
    buffer_size_seconds: float = 2.0  # Size of audio buffer before sending for transcription