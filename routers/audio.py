"""
Audio processing endpoints for Axon AI Tutor
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Optional

from models.api_models import AudioTranscriptionResponse
from core.memory import get_or_create_conversation
from processors.audio_processor import transcribe_audio, add_transcription_to_conversation
from core.state import conversations

# Create router
router = APIRouter()

@router.post("/transcribe", response_model=AudioTranscriptionResponse)
async def process_audio(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    language: str = Form("en"),
    model_size: str = Form("base"),
    use_groq: bool = Form(False)
):
    """Process audio file and add to conversation."""
    try:
        # Get or create conversation
        conversation_id, conversation_data = get_or_create_conversation(
            conversations, conversation_id
        )
        
        # Process audio file
        transcription = await transcribe_audio(
            file,
            language=language,
            model_size=model_size,
            use_groq=use_groq
        )
        
        # Add to conversation if it exists
        await add_transcription_to_conversation(transcription, conversation_data)
        
        return AudioTranscriptionResponse(
            text=transcription,
            conversation_id=conversation_id,
            language=language
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")