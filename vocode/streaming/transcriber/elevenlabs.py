import asyncio
import audioop
import io
import requests
from typing import Optional

import numpy as np
from loguru import logger

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import (
    Transcription,
    TimeEndpointingConfig,
    PunctuationEndpointingConfig,
    ElevenLabsTranscriberConfig,
)
from vocode.streaming.transcriber.base_transcriber import BaseAsyncTranscriber

class ElevenLabsTranscriber(BaseAsyncTranscriber[ElevenLabsTranscriberConfig]):
    def __init__(
        self,
        transcriber_config: ElevenLabsTranscriberConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = api_key or getenv("ELEVEN_LABS_API_KEY")
        if not self.api_key:
            raise Exception(
                "Please set ELEVEN_LABS_API_KEY environment variable or pass it as a parameter"
            )
        
        self._ended = False
        self.buffer = bytearray()
        self.audio_cursor = 0
        self.model_id = self.transcriber_config.model_id
        
        # Configure endpointing if provided
        if isinstance(
            self.transcriber_config.endpointing_config,
            (TimeEndpointingConfig, PunctuationEndpointingConfig),
        ):
            self.time_cutoff_seconds = self.transcriber_config.endpointing_config.time_cutoff_seconds
        else:
            self.time_cutoff_seconds = 0.4  # Default value
            
        self.last_transcription_time = 0
        
    async def ready(self):
        return True
    
    async def _run_loop(self):
        await self.process()
    
    def send_audio(self, chunk):
        # Convert mulaw to linear if needed
        if self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
            sample_width = 1
            if isinstance(chunk, np.ndarray):
                chunk = chunk.astype(np.int16)
                chunk = chunk.tobytes()
            chunk = audioop.ulaw2lin(chunk, sample_width)
        
        self.buffer.extend(chunk)
        
        # Process buffer when it reaches the configured size
        if (
            len(self.buffer) / (2 * self.transcriber_config.sampling_rate)
        ) >= self.transcriber_config.buffer_size_seconds:
            self.consume_nonblocking(self.buffer)
            self.buffer = bytearray()
    
    async def terminate(self):
        self._ended = True
        await super().terminate()
    
    async def process(self):
        self.audio_cursor = 0
        
        while not self._ended:
            try:
                chunk = await self.input_audio_queue.get()
                if chunk is None:
                    continue

                # Convert MULAW to PCM if needed
                if self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
                    chunk = audioop.ulaw2lin(chunk, 2)  # Convert to 16-bit PCM

                # Add to buffer
                self.buffer.extend(chunk)
                
                # Process buffer when it reaches the configured size
                if (
                    len(self.buffer) / (2 * self.transcriber_config.sampling_rate)
                ) >= self.transcriber_config.buffer_size_seconds:
                    self.consume_nonblocking(self.buffer)
                    self.buffer = bytearray()
                
                # Send to ElevenLabs for transcription using requests
                try:
                    url = "https://api.elevenlabs.io/v1/speech-to-text"
                    headers = {"xi-api-key": self.api_key}
                    
                    files = {"file": ("audio.wav", io.BytesIO(self.buffer), "audio/wav")}
                    params = {"model_id": self.model_id}
                    
                    # Use asyncio to run the request in a thread pool
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: requests.post(url, headers=headers, files=files, params=params)
                    )
                    
                    if response.status_code == 200:
                        response_json = response.json()
                        transcription_text = response_json.get("text", "")
                        
                        if transcription_text:
                            # Determine if this should be a final transcription
                            is_final = False
                            if transcription_text.strip().endswith(('.', '!', '?')):
                                is_final = True
                            
                            # Create and send the transcription
                            self.produce_nonblocking(
                                Transcription(
                                    message=transcription_text,
                                    confidence=0.9,  # ElevenLabs doesn't provide confidence scores
                                    is_final=is_final
                                )
                            )
                            
                            # Reset buffer if final
                            if is_final:
                                self.buffer = bytearray()
                    else:
                        logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                    
                except Exception as e:
                    logger.error(f"Error in ElevenLabs transcription: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ElevenLabs transcriber process: {e}")