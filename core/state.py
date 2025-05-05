"""
Shared state management for Axon AI Tutor
"""
from typing import Dict, Any

# In-memory storage for conversations (replace with database in production)
conversations: Dict[str, Dict[str, Any]] = {} 