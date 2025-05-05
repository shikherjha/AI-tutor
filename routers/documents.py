"""
Document handling endpoints for Axon AI Tutor
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Optional, List

from models.api_models import FileUploadResponse, ConversationRequest, ConversationResponse
from core.memory import get_or_create_conversation
from processors.document_processor import DocumentProcessor
from core.state import conversations

# Create router
router = APIRouter()

# Initialize document processor
doc_processor = DocumentProcessor()

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """Upload a file to be used in the tutoring session."""
    try:
        # Get or create conversation
        conversation_id, conversation_data = get_or_create_conversation(
            conversations, conversation_id
        )
        
        # Process the file
        file_info = await doc_processor.process_file(
            file, 
            conversation_data, 
            description
        )
        
        return FileUploadResponse(
            status="success",
            message=f"File {file.filename} uploaded and processed",
            conversation_id=conversation_id,
            filename=file.filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/list", response_model=List[str])
async def list_files(conversation_id: str):
    """List all files in a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    conversation_data = conversations[conversation_id]
    
    if "files" not in conversation_data or not conversation_data["files"]:
        return []
        
    return list(conversation_data["files"].keys())

@router.delete("/remove", response_model=ConversationResponse)
async def remove_file(conversation_id: str, filename: str):
    """Remove a file from a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    conversation_data = conversations[conversation_id]
    
    if ("files" not in conversation_data or 
        filename not in conversation_data["files"]):
        raise HTTPException(status_code=404, detail="File not found")
        
    # Remove the file
    file_info = conversation_data["files"][filename]
    doc_processor.cleanup_files({
        "files": {filename: file_info}
    })
    
    # Remove from conversation data
    del conversation_data["files"][filename]
    
    return ConversationResponse(
        status="success",
        message=f"File {filename} removed"
    )