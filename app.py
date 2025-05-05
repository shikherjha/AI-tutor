"""
Main FastAPI application file for Axon AI Tutor
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv

# Import routers
from routers import tutor, documents, audio
from core.state import conversations

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Axon AI - AI Tutor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route redirects to API docs
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

# Include routers
app.include_router(tutor.router, prefix="/api/tutor", tags=["tutor"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(audio.router, prefix="/api/audio", tags=["audio"])

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    from agents.mcp_agent import cleanup_mcp_sessions
    await cleanup_mcp_sessions(conversations)

if __name__ == "__main__":
    import uvicorn
    
    # Check if running directly (interactive mode)
    if os.environ.get("INTERACTIVE", "false").lower() == "true":
        from core.interactive import run_interactive_chat
        import asyncio
        asyncio.run(run_interactive_chat())
    else:
        # Run as API server
        uvicorn.run(app, host="127.0.0.1", port=8000)