"""
Translation utilities for Axon AI Tutor
"""
import json
import httpx
from typing import Optional
from fastapi import HTTPException

from config.settings import GOOGLE_CLOUD_API_KEY
from core.rate_limiter import translation_rate_limiter

async def translate_text(
    text: str, 
    source_language: Optional[str] = None, 
    target_language: str = "en"
) -> str:
    """
    Translate text using Google Cloud Translation API
    
    Args:
        text: Text to translate
        source_language: Source language code (auto-detect if None)
        target_language: Target language code
        
    Returns:
        Translated text
    """
    if not GOOGLE_CLOUD_API_KEY:
        raise HTTPException(status_code=500, detail="Translation API key not configured")
    
    # Wait for rate limit if needed
    await translation_rate_limiter.wait_if_needed()
    
    try:
        # Google Cloud Translation API endpoint
        url = "https://translation.googleapis.com/language/translate/v2"
        
        # Prepare request
        payload = {
            "q": text,
            "target": target_language,
            "key": GOOGLE_CLOUD_API_KEY
        }
        
        # Add source language if specified
        if source_language:
            payload["source"] = source_language
        
        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=payload)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Translation API error: {response.text}"
                )
                
            # Parse response
            result = response.json()
            
            if "data" in result and "translations" in result["data"]:
                translation = result["data"]["translations"][0]["translatedText"]
                return translation
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Invalid response from translation API"
                )
                
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Translation request error: {str(e)}")

async def detect_language(text: str) -> str:
    """
    Detect the language of text using Google Cloud Translation API
    
    Args:
        text: Text to detect language for
        
    Returns:
        Language code (e.g., 'en', 'es', 'fr')
    """
    if not GOOGLE_CLOUD_API_KEY:
        raise HTTPException(status_code=500, detail="Translation API key not configured")
    
    # Wait for rate limit if needed
    await translation_rate_limiter.wait_if_needed()
    
    try:
        # Google Cloud Translation API language detection endpoint
        url = f"https://translation.googleapis.com/language/translate/v2/detect?key={GOOGLE_CLOUD_API_KEY}"
        
        # Prepare request
        payload = {"q": text}
        
        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=payload)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Language detection API error: {response.text}"
                )
                
            # Parse response
            result = response.json()
            
            if "data" in result and "detections" in result["data"]:
                detection = result["data"]["detections"][0][0]
                language_code = detection["language"]
                return language_code
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Invalid response from language detection API"
                )
                
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Language detection request error: {str(e)}")