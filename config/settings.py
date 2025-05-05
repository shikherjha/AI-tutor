"""
Settings and configuration for Axon AI Tutor
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = Path(os.environ.get("TEMP_DIR", str(BASE_DIR / "tmp" / "axon_ai")))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Audio processing
AUDIO_UPLOAD_DIR = Path(os.environ.get("AUDIO_UPLOAD_DIR", BASE_DIR / "audio_uploads"))
AUDIO_UPLOAD_DIR.mkdir(exist_ok=True)

# API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
WOLFRAM_ALPHA_APPID = os.environ.get("WOLFRAM_ALPHA_APPID")
GOOGLE_CLOUD_API_KEY = os.environ.get("GOOGLE_CLOUD_API_KEY")

# LLM Settings
DEFAULT_LLM_PROVIDER = os.environ.get("DEFAULT_LLM_PROVIDER", "groq")
DEFAULT_LLM_MODEL = os.environ.get("DEFAULT_LLM_MODEL", "qwen-qwq-32b")
DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", "0.7"))

# Rate Limits
DDG_CALLS_PER_MINUTE = int(os.environ.get("DDG_CALLS_PER_MINUTE", "5"))
TAVILY_CALLS_PER_MINUTE = int(os.environ.get("TAVILY_CALLS_PER_MINUTE", "10"))
TRANSLATION_CALLS_PER_MINUTE = int(os.environ.get("TRANSLATION_CALLS_PER_MINUTE", "20"))

# MCP Agent Settings
MAX_MCP_STEPS = int(os.environ.get("MAX_MCP_STEPS", "15"))
MCP_CONFIG_PATH = os.environ.get("MCP_CONFIG_PATH", BASE_DIR / "config" / "mcp_config.json")

# Default language
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "en")