"""
LLM utilities for Axon AI Tutor
"""
from typing import Optional
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from config.settings import (
    GROQ_API_KEY,
    OPENAI_API_KEY,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE
)

def get_llm(provider: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None) -> BaseChatModel:
    """
    Initialize an LLM based on provider and model.
    
    Args:
        provider: LLM provider (groq, openai)
        model: Model name to use
        temperature: Temperature for generation
        
    Returns:
        An initialized LLM
    """
    provider = provider or DEFAULT_LLM_PROVIDER
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    
    if provider == "groq":
        return ChatGroq(
            model_name=model or DEFAULT_LLM_MODEL,
            temperature=temperature,
            api_key=GROQ_API_KEY
        )
    elif provider == "openai":
        return ChatOpenAI(
            model_name=model or "gpt-4-turbo",
            temperature=temperature,
            api_key=OPENAI_API_KEY
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}") 