# Voice Cloning Agent

An intelligent voice cloning agent that can clone, synthesize, and convert voices using state-of-the-art deep learning models.

## Features

- 🎤 **Voice Cloning**: Clone voices with minimal sample audio (5-30 seconds)
- 🔊 **Voice Synthesis**: Generate speech in cloned voices
- 🎯 **Speaker Embedding**: Extract and manage speaker profiles
- 🌐 **Multi-language Support**: Support for multiple languages
- ⚡ **Real-time Processing**: Fast inference for quick results
- 🔄 **Voice Conversion**: Convert one voice to another
- 📊 **Quality Metrics**: Voice quality assessment and validation

## Architecture

```
voice-cloning-agent/
├── src/
│   ├── agent/              # Main agent logic
│   │   ├── __init__.py
│   │   ├── voice_agent.py  # Core agent implementation
│   │   └── models.py       # Model management
│   ├── models/             # Deep learning models
│   │   ├── encoder.py      # Speaker encoder
│   │   ├── vocoder.py      # Audio vocoder
│   │   └── synthesizer.py  # TTS synthesizer
│   ├── utils/              # Utility functions
│   │   ├── audio.py        # Audio processing
│   │   ├── config.py       # Configuration management
│   │   └── logger.py       # Logging setup
│   └── api/                # REST API endpoints
│       └── server.py       # FastAPI server
├── data/
│   ├── models/             # Pre-trained models
│   └── samples/            # Sample audio files
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
├── config.yaml            # Configuration file
└── main.py                # Entry point
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)
- ffmpeg

### Setup

```bash
# Clone the repository
git clone https://github.com/hatafRakshi07/voice-cloning-agent.git
cd voice-cloning-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python src/utils/download_models.py
```

## Quick Start

### Basic Usage

```python
from src.agent.voice_agent import VoiceAgent

# Initialize agent
agent = VoiceAgent()

# Clone a voice from sample audio
voice_profile = agent.clone_voice(
    audio_path="path/to/sample_audio.wav",
    speaker_name="John Doe"
)

# Generate speech in cloned voice
agent.synthesize_speech(
    text="Hello, this is a cloned voice!",
    voice_profile=voice_profile,
    output_path="output.wav"
)
```

### CLI Usage

```bash
# Clone a voice
python main.py clone sample_audio.wav "John Doe"

# Synthesize speech
python main.py synthesize "Hello world" john-doe-0

# Convert voice
python main.py convert source.wav target-voice-id

# List all voices
python main.py list

# Start API server
python main.py server --port 8000
```

### API Usage

```bash
# Start the server
python main.py server

# Clone a voice (POST request)
curl -X POST http://localhost:8000/api/clone \
  -F "audio=@sample.wav" \
  -F "speaker_name=John"

# Synthesize speech
curl -X POST http://localhost:8000/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "voice_id": "john-doe-0"
  }'
```

## API Endpoints

### Clone Voice
```
POST /api/clone
```
**Parameters:**
- `audio` (file): Sample audio file (.wav, .mp3)
- `speaker_name` (string): Name for the cloned voice

### Synthesize Speech
```
POST /api/synthesize
```
**Parameters:**
- `text` (string): Text to synthesize
- `voice_id` (string): ID of cloned voice
- `language` (string): Language code (default: "en")
- `speed` (float): Speech speed multiplier (default: 1.0)

### List Voices
```
GET /api/voices
```

### Get Voice Details
```
GET /api/voices/{voice_id}
```

### Delete Voice
```
DELETE /api/voices/{voice_id}
```

### Convert Voice
```
POST /api/convert
```
**Parameters:**
- `audio` (file): Source audio file
- `target_voice_id` (string): Target voice ID
- `speed` (float): Speed multiplier (default: 1.0)

## Configuration

Edit `config.yaml` to customize models, audio settings, and processing parameters.

## Supported Models

### Encoder Models
- **ResNet Speaker Encoder**: Lightweight speaker embedding extraction
- **VGGVox2**: Advanced speaker verification

### Synthesizer Models
- **Tacotron2**: Sequence-to-sequence TTS
- **Glow-TTS**: Fast, high-quality TTS
- **FastSpeech2**: Real-time TTS synthesis

### Vocoder Models
- **HiFi-GAN**: High-fidelity neural vocoder
- **WaveGlow**: Flow-based vocoder

## Performance

- **Voice Cloning**: 2-5 seconds (GPU) / 10-15 seconds (CPU)
- **Speech Synthesis**: Real-time or faster (depending on text length)
- **Voice Conversion**: 1-2 seconds per second of audio

## Hardware Requirements

### Minimum
- CPU: 4 cores
- RAM: 8GB
- Storage: 2GB

### Recommended
- GPU: NVIDIA RTX 2080 or better
- RAM: 16GB
- Storage: 10GB SSD

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for educational and legitimate purposes only. Users are responsible for ensuring they have proper consent when cloning voices. Unauthorized voice cloning may violate laws and regulations in your jurisdiction.

## Support

For issues, questions, or suggestions, please open an [issue](https://github.com/hatafRakshi07/voice-cloning-agent/issues) on GitHub.

---

**Created with ❤️ for voice synthesis enthusiasts**
