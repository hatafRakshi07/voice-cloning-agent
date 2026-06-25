"""Main Voice Cloning Agent"""

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
    Main voice cloning agent that orchestrates voice cloning,
    synthesis, and conversion operations
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Voice Cloning Agent
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config() if not config_path else self._load_config(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.get("processing.device") == "cuda"
            else "cpu"
        )
        
        logger.info(f"Voice Agent initialized on device: {self.device}")
        
        # Initialize models (will be loaded on demand)
        self.encoder = None
        self.synthesizer = None
        self.vocoder = None
        
        # Voice storage
        self.voices: Dict[str, Dict[str, Any]] = {}

    def _load_config(self, config_path: str):
        """Load configuration from file"""
        from src.utils.config import Config
        return Config(config_path)

    def clone_voice(
        self,
        audio_path: str,
        speaker_name: str,
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Clone a voice from audio sample
        
        Args:
            audio_path: Path to audio file
            speaker_name: Name for the cloned voice
            duration: Duration to use from audio (uses full audio if None)
            
        Returns:
            Voice profile dictionary
        """
        logger.info(f"Cloning voice from {audio_path} for speaker: {speaker_name}")
        
        try:
            # Load audio
            audio = self._load_audio(audio_path)
            
            # Extract speaker embedding
            embedding = self._extract_embedding(audio)
            
            # Create voice profile
            voice_id = f"{speaker_name.lower().replace(' ', '_')}-{len(self.voices)}"
            voice_profile = {
                "voice_id": voice_id,
                "speaker_name": speaker_name,
                "embedding": embedding,
                "source_audio": audio_path,
                "duration": len(audio) / self.config.get("audio.sample_rate")
            }
            
            # Store voice profile
            self.voices[voice_id] = voice_profile
            
            logger.info(f"Voice cloned successfully: {voice_id}")
            return voice_profile
            
        except Exception as e:
            logger.error(f"Failed to clone voice: {e}")
            raise

    def synthesize_speech(
        self,
        text: str,
        voice_profile: Dict[str, Any],
        output_path: Optional[str] = None,
        speed: float = 1.0
    ) -> np.ndarray:
        """
        Synthesize speech in cloned voice
        
        Args:
            text: Text to synthesize
            voice_profile: Voice profile dictionary
            output_path: Path to save audio (optional)
            speed: Speech speed multiplier
            
        Returns:
            Generated audio waveform
        """
        logger.info(f"Synthesizing speech: '{text[:50]}...' with voice: {voice_profile['speaker_name']}")
        
        try:
            # Generate mel-spectrogram from text
            mel_spec = self._text_to_mel(text)
            
            # Convert mel-spec to waveform using vocoder
            audio = self._mel_to_waveform(mel_spec, voice_profile)
            
            # Adjust speed if needed
            if speed != 1.0:
                audio = self._change_speed(audio, speed)
            
            # Save if output path provided
            if output_path:
                self._save_audio(audio, output_path)
                logger.info(f"Audio saved to {output_path}")
            
            return audio
            
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {e}")
            raise

    def convert_voice(
        self,
        source_audio_path: str,
        target_voice_profile: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Convert voice from source to target
        
        Args:
            source_audio_path: Path to source audio
            target_voice_profile: Target voice profile
            output_path: Path to save converted audio
            
        Returns:
            Converted audio waveform
        """
        logger.info(f"Converting voice to: {target_voice_profile['speaker_name']}")
        
        try:
            # Load source audio
            audio = self._load_audio(source_audio_path)
            
            # Extract content and convert
            converted = self._voice_conversion(audio, target_voice_profile)
            
            # Save if output path provided
            if output_path:
                self._save_audio(converted, output_path)
                logger.info(f"Converted audio saved to {output_path}")
            
            return converted
            
        except Exception as e:
            logger.error(f"Failed to convert voice: {e}")
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
            logger.info(f"Voice deleted: {voice_id}")
            return True
        return False

    # ========== Private Methods ==========

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file"""
        import librosa
        
        audio, _ = librosa.load(
            audio_path,
            sr=self.config.get("audio.sample_rate")
        )
        return audio

    def _save_audio(self, audio: np.ndarray, output_path: str) -> None:
        """Save audio file"""
        import soundfile as sf
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, self.config.get("audio.sample_rate"))

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio"""
        # Placeholder: In production, this would use a speaker encoder model
        logger.debug("Extracting speaker embedding...")
        return np.random.randn(self.config.get("cloning.embedding_dim")).astype(np.float32)

    def _text_to_mel(self, text: str) -> np.ndarray:
        """Convert text to mel-spectrogram"""
        # Placeholder: In production, this would use TTS models
        logger.debug("Converting text to mel-spectrogram...")
        n_mels = self.config.get("audio.n_mels")
        return np.random.randn(n_mels, 100).astype(np.float32)

    def _mel_to_waveform(self, mel_spec: np.ndarray, voice_profile: Dict[str, Any]) -> np.ndarray:
        """Convert mel-spectrogram to waveform using vocoder"""
        # Placeholder: In production, this would use a vocoder model
        logger.debug("Converting mel-spec to waveform...")
        return np.random.randn(22050 * 4).astype(np.float32)

    def _voice_conversion(self, audio: np.ndarray, target_voice: Dict[str, Any]) -> np.ndarray:
        """Convert voice"""
        # Placeholder: In production, this would use voice conversion models
        logger.debug("Converting voice...")
        return audio

    def _change_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Change audio speed"""
        if speed == 1.0:
            return audio
        
        return audio[::int(1/speed)] if speed > 1.0 else np.repeat(audio, int(speed))


if __name__ == "__main__":
    # Example usage
    agent = VoiceAgent()
    logger.info("Voice agent initialized successfully")
