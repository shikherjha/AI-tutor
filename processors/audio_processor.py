"""
Audio processing utilities for Axon AI Tutor
"""
import os
import uuid
from typing import Optional, Dict, Any
from fastapi import UploadFile, HTTPException
from pathlib import Path

import whisper  # You'll need to install this: pip install openai-whisper
from config.settings import AUDIO_UPLOAD_DIR, GROQ_API_KEY

# For advanced audio processing
import httpx

# Initialize whisper model (load lazily)
whisper_model = None

def get_whisper_model(model_size: str = "base"):
    """
    Get or initialize whisper model
    
    Args:
        model_size: Size of the model to use ('tiny', 'base', 'small', 'medium', 'large')
        
    Returns:
        Initialized whisper model
    """
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model(model_size)  
    return whisper_model

async def transcribe_audio(
    file: UploadFile, 
    language: str = "en",
    model_size: str = "base",
    use_groq: bool = False
) -> str:
    """
    Process an audio file and return transcription
    
    Args:
        file: Audio file to transcribe
        language: Language code (e.g., 'en', 'es', 'fr')
        model_size: Whisper model size
        use_groq: Whether to use Groq API for transcription
        
    Returns:
        Transcribed text
    """
    # Generate unique filename to avoid collisions
    file_path = AUDIO_UPLOAD_DIR / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    
    try:
        # Save the uploaded audio file temporarily
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        if use_groq and GROQ_API_KEY:
            # Use Groq API for more advanced transcription via their Whisper implementation
            async with httpx.AsyncClient() as client:
                with open(file_path, "rb") as audio_file:
                    files = {"file": (file.filename, audio_file)}
                    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
                    
                    response = await client.post(
                        "https://api.groq.com/v1/audio/transcriptions",
                        files=files,
                        headers=headers,
                        data={
                            "model": "whisper-large-v3",
                            "language": language
                        },
                        timeout=60.0  # Allow up to 60 seconds for large files
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        transcription = result.get("text", "")
                    else:
                        raise HTTPException(
                            status_code=response.status_code, 
                            detail=f"Groq API error: {response.text}"
                        )
        else:
            # Use local Whisper model
            model = get_whisper_model(model_size)
            result = model.transcribe(str(file_path), language=language)
            transcription = result["text"]
        
        return transcription
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")
    
    finally:
        # Clean up
        if file_path.exists():
            os.remove(file_path)

async def add_transcription_to_conversation(
    transcription: str,
    conversation_data: Dict[str, Any]
) -> None:
    """
    Add transcription to conversation memory
    
    Args:
        transcription: Transcribed text
        conversation_data: Conversation data dictionary
    """
    if "memory" in conversation_data:
        conversation_data["memory"].chat_memory.add_user_message(transcription)