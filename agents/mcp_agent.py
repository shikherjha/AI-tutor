"""
MCP Agent implementation for Axon AI Tutor
"""
import json
from typing import Dict, Any, Optional, List
from mcp_use import MCPAgent, MCPClient
from langchain_core.language_models import BaseChatModel

from config.settings import MCP_CONFIG_PATH, MAX_MCP_STEPS

async def get_mcp_agent(
    llm: BaseChatModel, 
    memory_enabled: bool = True,
    max_steps: int = MAX_MCP_STEPS
) -> tuple[MCPAgent, MCPClient]:
    """
    Initialize MCP agent with specified tools.
    
    Args:
        llm: Language model to use
        memory_enabled: Whether to enable memory for the agent
        max_steps: Maximum number of steps the agent can take
        
    Returns:
        Tuple of (agent, client)
    """
    # Load MCP config
    with open(MCP_CONFIG_PATH, 'r') as f:
        mcp_config = json.load(f)
    
    # Initialize client
    client = await MCPClient.from_config(mcp_config)
    
    # Create agent
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=max_steps,
        memory_enabled=memory_enabled,
    )
    
    # Add custom system message to ensure educational focus
    system_message = """You are an advanced AI tutor designed to help with education.
    Your primary goals are to:
    1. Explain complex topics clearly and accurately
    2. Answer questions thoroughly with examples
    3. Break down difficult concepts into manageable pieces
    4. Use web search and other tools to provide up-to-date information
    5. Help with problem-solving through guided reasoning
    
    When using tools, prioritize educational resources and reliable sources.
    Always explain your reasoning process to help the student understand how to approach similar problems.
    """
    
    agent.set_system_message(system_message)
    return agent, client

def extract_sources_from_agent_response(response: str) -> List[str]:
    """
    Extract sources from agent response if available.
    
    Args:
        response: Response from the agent
        
    Returns:
        List of source URLs
    """
    sources = []
    if "Source:" in response:
        parts = response.split("Source:")
        for part in parts[1:]:
            source = part.strip().split("\n")[0]
            sources.append(source)
    return sources

async def cleanup_mcp_sessions(conversations: Dict[str, Dict[str, Any]]) -> None:
    """
    Clean up MCP sessions on shutdown
    
    Args:
        conversations: Dictionary of all conversations
    """
    for conv_id, conv_data in conversations.items():
        if conv_data.get("client"):
            await conv_data["client"].close_all_sessions()