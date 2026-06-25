"""FastAPI Server for Voice Cloning Agent"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import logging

from src.agent.voice_agent import VoiceAgent
from src.utils.config import get_config

logger = logging.getLogger(__name__)

# Initialize app and agent
app = FastAPI(
    title="Voice Cloning Agent API",
    description="API for voice cloning and synthesis",
    version="1.0.0"
)

agent = VoiceAgent()
config = get_config()


# ========== Data Models ==========

class VoiceProfile(BaseModel):
    """Voice profile response model"""
    voice_id: str
    speaker_name: str
    duration: float
    quality_score: Optional[float] = None


class SynthesizeRequest(BaseModel):
    """Speech synthesis request"""
    text: str
    voice_id: str
    language: str = "en"
    speed: float = 1.0


class VoiceConversionRequest(BaseModel):
    """Voice conversion request"""
    target_voice_id: str
    speed: float = 1.0


# ========== Health Check ==========

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(agent.device),
        "voices_loaded": len(agent.list_voices())
    }


# ========== Voice Management ==========

@app.post("/api/clone", response_model=VoiceProfile)
async def clone_voice(
    audio: UploadFile = File(...),
    speaker_name: str = "Unknown"
):
    """
    Clone a voice from audio file
    
    Args:
        audio: Audio file (WAV, MP3)
        speaker_name: Name for the cloned voice
        
    Returns:
        Voice profile with ID and metadata
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{audio.filename}"
        with open(temp_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Clone voice
        voice_profile = agent.clone_voice(temp_path, speaker_name)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return VoiceProfile(
            voice_id=voice_profile["voice_id"],
            speaker_name=voice_profile["speaker_name"],
            duration=voice_profile["duration"]
        )
        
    except Exception as e:
        logger.error(f"Failed to clone voice: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/voices")
async def list_voices():
    """Get all cloned voices"""
    voices = []
    for voice_id in agent.list_voices():
        profile = agent.get_voice(voice_id)
        voices.append({
            "voice_id": voice_id,
            "speaker_name": profile["speaker_name"],
            "duration": profile["duration"]
        })
    return {"voices": voices}


@app.get("/api/voices/{voice_id}")
async def get_voice(voice_id: str):
    """Get specific voice profile"""
    profile = agent.get_voice(voice_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    return {
        "voice_id": voice_id,
        "speaker_name": profile["speaker_name"],
        "duration": profile["duration"]
    }


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a voice profile"""
    if not agent.delete_voice(voice_id):
        raise HTTPException(status_code=404, detail="Voice not found")
    
    return {"message": f"Voice {voice_id} deleted successfully"}


# ========== Synthesis ==========

@app.post("/api/synthesize")
async def synthesize_speech(request: SynthesizeRequest):
    """
    Synthesize speech in cloned voice
    
    Args:
        request: Synthesis request with text and voice ID
        
    Returns:
        Path to generated audio file
    """
    try:
        # Get voice profile
        voice_profile = agent.get_voice(request.voice_id)
        if not voice_profile:
            raise HTTPException(status_code=404, detail="Voice not found")
        
        # Generate output filename
        import time
        output_filename = f"synthesis_{request.voice_id}_{int(time.time())}.wav"
        output_path = os.path.join(config.get("storage.output_path"), output_filename)
        
        # Synthesize
        agent.synthesize_speech(
            text=request.text,
            voice_profile=voice_profile,
            output_path=output_path,
            speed=request.speed
        )
        
        return {
            "filename": output_filename,
            "url": f"/outputs/{output_filename}",
            "voice_id": request.voice_id
        }
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ========== Voice Conversion ==========

@app.post("/api/convert")
async def convert_voice(
    audio: UploadFile = File(...),
    target_voice_id: str = "",
    speed: float = 1.0
):
    """
    Convert voice to target speaker
    
    Args:
        audio: Source audio file
        target_voice_id: Target voice ID
        speed: Speed multiplier
        
    Returns:
        Path to converted audio file
    """
    try:
        # Get target voice profile
        target_profile = agent.get_voice(target_voice_id)
        if not target_profile:
            raise HTTPException(status_code=404, detail="Target voice not found")
        
        # Save uploaded file temporarily
        temp_path = f"temp_{audio.filename}"
        with open(temp_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Generate output filename
        import time
        output_filename = f"converted_{target_voice_id}_{int(time.time())}.wav"
        output_path = os.path.join(config.get("storage.output_path"), output_filename)
        
        # Convert voice
        agent.convert_voice(temp_path, target_profile, output_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "filename": output_filename,
            "url": f"/outputs/{output_filename}",
            "target_voice_id": target_voice_id
        }
        
    except Exception as e:
        logger.error(f"Voice conversion failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ========== File Serving ==========

@app.get("/outputs/{filename}")
async def download_output(filename: str):
    """Download generated audio file"""
    output_dir = config.get("storage.output_path")
    filepath = os.path.join(output_dir, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filepath, media_type="audio/wav")


# ========== Root ==========

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Voice Cloning Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    
    uvicorn.run(app, host=host, port=port)
