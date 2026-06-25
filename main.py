#!/usr/bin/env python3
"""Main Entry Point for Voice Cloning Agent"""

import argparse
import logging
from pathlib import Path

from src.agent.voice_agent import VoiceAgent
from src.utils.logger import setup_logger


def setup_directories():
    """Create necessary directories"""
    dirs = [
        "data/models",
        "data/voices",
        "outputs",
        "logs"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Voice Cloning Agent - Clone, synthesize, and convert voices"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Clone command
    clone_parser = subparsers.add_parser("clone", help="Clone a voice")
    clone_parser.add_argument("audio_path", help="Path to audio file")
    clone_parser.add_argument("speaker_name", help="Name for the cloned voice")
    
    # Synthesize command
    synth_parser = subparsers.add_parser("synthesize", help="Synthesize speech")
    synth_parser.add_argument("text", help="Text to synthesize")
    synth_parser.add_argument("voice_id", help="Voice ID to use")
    synth_parser.add_argument("--output", "-o", help="Output audio path")
    synth_parser.add_argument("--speed", "-s", type=float, default=1.0, help="Speech speed")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert voice")
    convert_parser.add_argument("audio_path", help="Path to source audio")
    convert_parser.add_argument("target_voice_id", help="Target voice ID")
    convert_parser.add_argument("--output", "-o", help="Output audio path")
    
    # List command
    subparsers.add_parser("list", help="List all cloned voices")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    server_parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(
        "voice_agent",
        log_level="INFO",
        log_file="logs/voice_agent.log"
    )
    
    # Setup directories
    setup_directories()
    
    # Initialize agent
    agent = VoiceAgent()
    
    # Execute commands
    if args.command == "clone":
        logger.info(f"Cloning voice from {args.audio_path}")
        profile = agent.clone_voice(args.audio_path, args.speaker_name)
        print(f"✓ Voice cloned successfully!")
        print(f"  Voice ID: {profile['voice_id']}")
        print(f"  Duration: {profile['duration']:.2f}s")
        
    elif args.command == "synthesize":
        logger.info(f"Synthesizing: '{args.text}'")
        voice = agent.get_voice(args.voice_id)
        if not voice:
            print(f"✗ Voice not found: {args.voice_id}")
            return
        
        output_path = args.output or f"output_{args.voice_id}.wav"
        agent.synthesize_speech(
            args.text,
            voice,
            output_path=output_path,
            speed=args.speed
        )
        print(f"✓ Speech synthesized successfully!")
        print(f"  Output: {output_path}")
        
    elif args.command == "convert":
        logger.info(f"Converting voice from {args.audio_path}")
        voice = agent.get_voice(args.target_voice_id)
        if not voice:
            print(f"✗ Voice not found: {args.target_voice_id}")
            return
        
        output_path = args.output or f"converted_{args.target_voice_id}.wav"
        agent.convert_voice(args.audio_path, voice, output_path=output_path)
        print(f"✓ Voice converted successfully!")
        print(f"  Output: {output_path}")
        
    elif args.command == "list":
        voices = agent.list_voices()
        if voices:
            print("Cloned Voices:")
            for voice_id in voices:
                voice = agent.get_voice(voice_id)
                print(f"  • {voice['speaker_name']} ({voice_id}) - {voice['duration']:.2f}s")
        else:
            print("No voices cloned yet.")
            
    elif args.command == "server":
        logger.info(f"Starting API server on {args.host}:{args.port}")
        from src.api.server import app
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
