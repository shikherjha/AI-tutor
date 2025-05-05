"""
Pydantic models for API requests and responses
"""
from typing import List, Optional
from pydantic import BaseModel

# Tutor query models
class TutorQuery(BaseModel):
    """Request model for asking questions to the AI tutor"""
    question: str
    conversation_id: Optional[str] = None
    language: Optional[str] = "english"
    use_tavily: Optional[bool] = True
    use_mcp: Optional[bool] = True
    use_langgraph: Optional[bool] = False

class TutorResponse(BaseModel):
    """Response model for AI tutor answers"""
    answer: str
    conversation_id: str
    sources: Optional[List[str]] = None
    language: Optional[str] = "english"

# File upload models
class FileUploadRequest(BaseModel):
    """Request model for file upload"""
    conversation_id: Optional[str] = None
    description: Optional[str] = None

class FileUploadResponse(BaseModel):
    """Response model for file upload"""
    status: str
    message: str
    conversation_id: str
    filename: str

# Audio processing models
class AudioTranscriptionRequest(BaseModel):
    """Request model for audio transcription"""
    conversation_id: Optional[str] = None
    language: Optional[str] = "en"
    model_size: Optional[str] = "base"
    use_groq: Optional[bool] = False

class AudioTranscriptionResponse(BaseModel):
    """Response model for audio transcription"""
    text: str
    conversation_id: str
    language: str

# Translation models
class TranslationRequest(BaseModel):
    """Request model for text translation"""
    text: str
    source_language: Optional[str] = None
    target_language: str = "en"

class TranslationResponse(BaseModel):
    """Response model for text translation"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str

# Conversation management models
class ConversationRequest(BaseModel):
    """Request model for conversation operations"""
    conversation_id: str

class ConversationResponse(BaseModel):
    """Response model for conversation operations"""
    status: str
    message: str