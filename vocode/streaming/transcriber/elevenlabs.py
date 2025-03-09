import asyncio
import audioop
import logging
import os
import queue
import time
from typing import Dict, List, Optional, Tuple

import aiohttp
from loguru import logger
from pydantic import Field

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import ElevenLabsTranscriberConfig
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber, Transcription

class ElevenLabsTranscriber(BaseTranscriber):
    def __init__(self, transcriber_config: ElevenLabsTranscriberConfig):
        super().__init__(transcriber_config)
        self.transcriber_config = transcriber_config
        self.api_key = transcriber_config.api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")
        
        # Initialize the audio buffer
        self.buffer = bytearray()
        self.buffer_size_seconds = 2.0  # Buffer 2 seconds of audio before sending
        self.buffer_size_bytes = int(self.transcriber_config.sampling_rate * 2 * self.buffer_size_seconds)  # 16-bit audio = 2 bytes per sample
        
        # Initialize state
        self._ended = False
        self.is_ready = True

    async def _transcribe_chunk(self, audio_chunk: bytes) -> Optional[str]:
        """Send audio chunk to ElevenLabs API and get transcription"""
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "audio/wav"  # ElevenLabs expects WAV format
        }
        
        # Create a simple WAV header for the PCM data
        # This is a minimal WAV header for 16-bit PCM mono audio
        sample_rate = self.transcriber_config.sampling_rate
        channels = 1
        bits_per_sample = 16
        
        # WAV header
        wav_header = bytearray()
        # RIFF header
        wav_header.extend(b'RIFF')
        wav_header.extend((36 + len(audio_chunk)).to_bytes(4, 'little'))  # File size - 8
        wav_header.extend(b'WAVE')
        # fmt chunk
        wav_header.extend(b'fmt ')
        wav_header.extend((16).to_bytes(4, 'little'))  # Subchunk1Size
        wav_header.extend((1).to_bytes(2, 'little'))  # AudioFormat (PCM)
        wav_header.extend((channels).to_bytes(2, 'little'))  # NumChannels
        wav_header.extend((sample_rate).to_bytes(4, 'little'))  # SampleRate
        wav_header.extend((sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))  # ByteRate
        wav_header.extend((channels * bits_per_sample // 8).to_bytes(2, 'little'))  # BlockAlign
        wav_header.extend((bits_per_sample).to_bytes(2, 'little'))  # BitsPerSample
        # data chunk
        wav_header.extend(b'data')
        wav_header.extend((len(audio_chunk)).to_bytes(4, 'little'))  # Subchunk2Size
        
        # Combine header and audio data
        wav_data = wav_header + audio_chunk
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=wav_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("text", "")
                    else:
                        error_text = await response.text()
                        logger.error(f"ElevenLabs API error: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logger.error(f"Error in ElevenLabs transcription: {e}")
            return None

    async def process(self):
        try:
            while not self._ended:
                try:
                    # Get audio chunk from the queue
                    chunk = await self.input_audio_queue.get()
                    if chunk is None:
                        continue

                    # Convert MULAW to PCM if needed
                    if self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
                        chunk = audioop.ulaw2lin(chunk, 2)  # Convert to 16-bit PCM

                    # Add to buffer
                    self.buffer.extend(chunk)
                    
                    # Process buffer when it reaches the desired size
                    if len(self.buffer) >= self.buffer_size_bytes:
                        # Extract the buffer content
                        audio_data = bytes(self.buffer)
                        self.buffer.clear()
                        
                        # Send for transcription
                        transcription = await self._transcribe_chunk(audio_data)
                        if transcription and transcription.strip():
                            await self.output_transcription_queue.put(
                                Transcription(
                                    message=transcription,
                                    confidence=1.0,  # ElevenLabs doesn't provide confidence scores
                                    is_final=True
                                )
                            )
                
                except asyncio.CancelledError:
                    logger.debug("ElevenLabs transcriber task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in ElevenLabs transcriber process: {e}")
                    
        except Exception as e:
            logger.error(f"Fatal error in ElevenLabs transcriber: {e}")
        finally:
            # Process any remaining audio in the buffer
            if len(self.buffer) > 0:
                try:
                    audio_data = bytes(self.buffer)
                    transcription = await self._transcribe_chunk(audio_data)
                    if transcription and transcription.strip():
                        await self.output_transcription_queue.put(
                            Transcription(
                                message=transcription,
                                confidence=1.0,
                                is_final=True
                            )
                        )
                except Exception as e:
                    logger.error(f"Error processing final buffer: {e}")

    async def terminate(self):
        self._ended = True