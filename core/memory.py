"""
Memory management for conversations
"""
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any, Optional

def create_conversation_memory() -> ConversationBufferMemory:
    """
    Create a new conversation memory buffer
    
    Returns:
        New conversation memory instance
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

def get_or_create_conversation(
    conversations: Dict[str, Dict[str, Any]], 
    conversation_id: Optional[str] = None
) -> tuple[str, Dict[str, Any]]:
    """
    Get an existing conversation or create a new one
    
    Args:
        conversations: Dictionary of all conversations
        conversation_id: ID of conversation to retrieve or None to create new
        
    Returns:
        Tuple of (conversation_id, conversation_data)
    """
    if conversation_id is None:
        # Create a new ID
        conversation_id = f"conv_{len(conversations) + 1}"
        
    # Initialize if not exists
    if conversation_id not in conversations:
        conversations[conversation_id] = {
            "memory": create_conversation_memory(),
            "agent": None,
            "client": None,
            "files": {}
        }
    
    return conversation_id, conversations[conversation_id]