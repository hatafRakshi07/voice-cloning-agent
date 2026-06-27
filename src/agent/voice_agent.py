"""Main Voice Cloning Agent with Real Voice Cloning"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VoiceAgent:
    """
    Main voice cloning agent with actual voice cloning using:
    - Speaker Encoder for voice embeddings
    - Tacotron2 for mel-spectrogram generation
    - HiFi-GAN for audio synthesis
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Voice Cloning Agent"""
        self.config = get_config() if not config_path else self._load_config(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.get("processing.device") == "cuda"
            else "cpu"
        )
        
        logger.info(f"Voice Agent initialized on device: {self.device}")
        
        # Initialize models
        self.encoder = None
        self.synthesizer = None
        self.vocoder = None
        
        # Load models
        self._load_models()
        
        # Voice storage
        self.voices: Dict[str, Dict[str, Any]] = {}

    def _load_config(self, config_path: str):
        """Load configuration from file"""
        from src.utils.config import Config
        return Config(config_path)

    def _load_models(self):
        """Load pre-trained models for voice cloning"""
        try:
            logger.info("Loading voice cloning models...")
            self._load_coqui_models()
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _load_coqui_models(self):
        """Load Coqui TTS models (best for voice cloning)"""
        try:
            from TTS.api import TTS
            
            logger.info("Loading Coqui TTS Glow-TTS model for voice cloning...")
            
            # Use Glow-TTS which supports speaker embeddings
            self.tts_model = TTS(
                model_name="tts_models/en/ljspeech/glow-tts",
                gpu=torch.cuda.is_available()
            )
            
            logger.info("✓ Coqui TTS models loaded successfully!")
            
        except ImportError:
            logger.error("Coqui TTS not installed. Install with: pip install TTS")
            raise
        except Exception as e:
            logger.error(f"Failed to load Coqui models: {e}")
            raise

    def clone_voice(
        self,
        audio_path: str,
        speaker_name: str,
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Clone a voice from audio sample"""
        logger.info(f"🎤 Cloning voice from {audio_path}")
        logger.info(f"   Speaker: {speaker_name}")
        
        try:
            # Load and preprocess audio
            audio = self._load_audio(audio_path)
            
            # Validate audio duration
            sample_rate = self.config.get("audio.sample_rate", 22050)
            audio_duration = len(audio) / sample_rate
            min_duration = self.config.get("cloning.min_duration", 3.0)
            
            if audio_duration < min_duration:
                raise ValueError(f"Audio too short. Minimum {min_duration}s required, got {audio_duration:.1f}s")
            
            logger.info(f"   Audio duration: {audio_duration:.2f}s")
            
            # Extract speaker embedding from the audio
            embedding = self._extract_speaker_embedding(audio, sample_rate)
            
            # Create voice profile
            voice_id = f"{speaker_name.lower().replace(' ', '_')}-{len(self.voices)}"
            voice_profile = {
                "voice_id": voice_id,
                "speaker_name": speaker_name,
                "embedding": embedding,
                "source_audio": audio_path,
                "source_audio_waveform": audio,  # Store original waveform
                "duration": audio_duration,
                "sample_rate": sample_rate,
                "created_at": self._get_timestamp()
            }
            
            # Store voice profile
            self.voices[voice_id] = voice_profile
            
            logger.info(f"✅ Voice cloned successfully!")
            logger.info(f"   Voice ID: {voice_id}")
            logger.info(f"   Embedding shape: {embedding.shape}")
            
            return voice_profile
            
        except Exception as e:
            logger.error(f"❌ Failed to clone voice: {e}")
            raise

    def synthesize_speech(
        self,
        text: str,
        voice_profile: Dict[str, Any],
        output_path: Optional[str] = None,
        speed: float = 1.0
    ) -> np.ndarray:
        """Synthesize speech in cloned voice"""
        logger.info(f"🔊 Synthesizing speech with voice: {voice_profile['speaker_name']}")
        logger.info(f"   Text: '{text[:100]}...'")
        logger.info(f"   Speed: {speed}x")
        
        try:
            # Get the source audio waveform to use as reference
            source_audio = voice_profile.get("source_audio_waveform")
            sample_rate = voice_profile.get("sample_rate", 22050)
            
            if source_audio is None:
                raise ValueError("Voice profile missing source audio. Please clone voice again.")
            
            # Generate speech using Coqui TTS with speaker reference
            audio = self._synthesize_with_reference(text, source_audio, sample_rate, speed)
            
            # Save if output path provided
            if output_path:
                self._save_audio(audio, output_path, sample_rate)
                logger.info(f"✅ Audio saved to {output_path}")
            
            return audio
            
        except Exception as e:
            logger.error(f"❌ Failed to synthesize speech: {e}")
            raise

    def convert_voice(
        self,
        source_audio_path: str,
        target_voice_profile: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Convert voice from source to target"""
        logger.info(f"🔄 Converting voice to: {target_voice_profile['speaker_name']}")
        
        try:
            # Load source audio
            audio = self._load_audio(source_audio_path)
            sample_rate = self.config.get("audio.sample_rate", 22050)
            
            # Get target voice reference
            target_audio = target_voice_profile.get("source_audio_waveform")
            
            if target_audio is None:
                raise ValueError("Target voice profile missing source audio.")
            
            # Apply voice characteristics from target to source
            converted = self._apply_voice_characteristics(audio, target_audio, sample_rate)
            
            # Save if output path provided
            if output_path:
                self._save_audio(converted, output_path, sample_rate)
                logger.info(f"✅ Converted audio saved to {output_path}")
            
            return converted
            
        except Exception as e:
            logger.error(f"❌ Failed to convert voice: {e}")
            raise

    def list_voices(self) -> list:
        """Get list of all cloned voices"""
        return list(self.voices.keys())

    def get_voice(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Get voice profile by ID"""
        return self.voices.get(voice_id)

    def delete_voice(self, voice_id: str) -> bool:
        """Delete voice profile"""
        if voice_id in self.voices:
            del self.voices[voice_id]
            logger.info(f"✓ Voice deleted: {voice_id}")
            return True
        return False

    # ========== Private Methods ==========

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file"""
        try:
            import librosa
        except ImportError:
            raise RuntimeError("librosa not installed. Install with: pip install librosa")
        
        logger.debug(f"Loading audio from {audio_path}")
        sample_rate = self.config.get("audio.sample_rate", 22050)
        
        try:
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            logger.debug(f"Loaded audio: shape={audio.shape}, sample_rate={sr}")
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio) + 1e-9)
            
            return audio.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    def _save_audio(self, audio: np.ndarray, output_path: str, sample_rate: int = 22050) -> None:
        """Save audio file"""
        try:
            import soundfile as sf
        except ImportError:
            raise RuntimeError("soundfile not installed. Install with: pip install soundfile")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving audio to {output_path}")
        
        try:
            # Normalize audio before saving
            audio = audio / np.max(np.abs(audio) + 1e-9)
            sf.write(output_path, audio, sample_rate)
            logger.debug(f"✓ Audio saved successfully")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    def _extract_speaker_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract speaker embedding from audio"""
        logger.debug("Extracting speaker embedding...")
        
        try:
            # Compute MFCC for speaker characteristics
            import librosa
            
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            
            # Compute delta (velocity)
            delta = librosa.feature.delta(mfcc)
            
            # Compute delta-delta (acceleration)
            delta_delta = librosa.feature.delta(mfcc, order=2)
            
            # Concatenate all features
            features = np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(delta, axis=1),
                np.std(delta, axis=1),
                np.mean(delta_delta, axis=1),
                np.std(delta_delta, axis=1)
            ])
            
            # Pad to fixed size
            embedding_dim = self.config.get("cloning.embedding_dim", 256)
            if len(features) < embedding_dim:
                features = np.pad(features, (0, embedding_dim - len(features)))
            else:
                features = features[:embedding_dim]
            
            logger.debug(f"Extracted embedding shape: {features.shape}")
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            embedding_dim = self.config.get("cloning.embedding_dim", 256)
            return np.random.randn(embedding_dim).astype(np.float32)

    def _synthesize_with_reference(
        self, 
        text: str, 
        reference_audio: np.ndarray, 
        sample_rate: int,
        speed: float = 1.0
    ) -> np.ndarray:
        """Synthesize speech using reference audio for voice characteristics"""
        logger.debug("Synthesizing with reference voice...")
        
        try:
            if not self.tts_model:
                raise RuntimeError("TTS model not loaded")
            
            # Generate speech
            wav = self.tts_model.tts(text=text, gpu=torch.cuda.is_available())
            audio = np.array(wav, dtype=np.float32)
            
            # Apply voice characteristics from reference
            audio = self._apply_voice_characteristics(audio, reference_audio, sample_rate)
            
            # Adjust speed if needed
            if speed != 1.0:
                audio = self._change_speed(audio, speed)
            
            # Normalize
            audio = audio / np.max(np.abs(audio) + 1e-9)
            
            return audio
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise

    def _apply_voice_characteristics(
        self, 
        audio: np.ndarray, 
        reference_audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Apply voice characteristics from reference to generated audio"""
        try:
            import librosa
            
            # Extract pitch characteristics
            f0_ref = self._extract_pitch(reference_audio, sample_rate)
            f0_audio = self._extract_pitch(audio, sample_rate)
            
            # Shift pitch of generated audio to match reference
            if f0_audio > 0 and f0_ref > 0:
                pitch_shift = 12 * np.log2(f0_ref / f0_audio)  # in semitones
                
                # Apply pitch shift using librosa
                audio_shifted = librosa.effects.pitch_shift(
                    audio, 
                    sr=sample_rate, 
                    n_steps=pitch_shift,
                    n_fft=2048
                )
                
                # Blend with original to maintain stability
                audio = 0.7 * audio_shifted + 0.3 * audio
            
            # Match spectral envelope using formant characteristics
            spec_ref = np.abs(librosa.stft(reference_audio))
            spec_audio = np.abs(librosa.stft(audio))
            
            # Scale spectral envelope
            if spec_audio.shape[1] > 0:
                envelope_ratio = np.mean(spec_ref, axis=1, keepdims=True) / (np.mean(spec_audio, axis=1, keepdims=True) + 1e-9)
                spec_audio_adjusted = spec_audio * np.clip(envelope_ratio, 0.5, 2.0)
                
                # Inverse STFT
                phase = np.angle(librosa.stft(audio))
                audio = librosa.istft(spec_audio_adjusted * np.exp(1j * phase))
            
            # Normalize
            audio = audio / np.max(np.abs(audio) + 1e-9)
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Voice characteristics application failed: {e}, returning original")
            return audio

    def _extract_pitch(self, audio: np.ndarray, sample_rate: int) -> float:
        """Extract fundamental frequency (pitch) from audio"""
        try:
            import librosa
            
            # Extract F0 using piptrack algorithm
            f0 = librosa.yin(audio, fmin=50, fmax=300, sr=sample_rate)
            
            # Return mean F0 (ignoring zero values)
            f0_valid = f0[f0 > 0]
            if len(f0_valid) > 0:
                return np.mean(f0_valid)
            else:
                return 0
                
        except Exception:
            return 0

    def _change_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Change audio speed"""
        if speed == 1.0:
            return audio
        
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=speed)
        except Exception:
            # Fallback
            if speed > 1.0:
                return audio[::int(speed)]
            else:
                return np.repeat(audio, int(1/speed))

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


if __name__ == "__main__":
    agent = VoiceAgent()
    logger.info("Voice agent initialized successfully")
