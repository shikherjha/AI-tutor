"""
Interactive console mode for Axon AI Tutor
"""
import asyncio
from dotenv import load_dotenv

from core.llm import get_llm
from agents.mcp_agent import get_mcp_agent
from core.rate_limiter import ddg_rate_limiter

async def run_interactive_chat():
    """Run an interactive chat session in the console"""
    load_dotenv()
    
    print("\n===== Interactive AI Tutor Chat =====")
    print("Type 'exit' to end the chat")
    print("Type 'clear' to clear conversation history")
    print("======================================\n")
    
    llm = get_llm()
    agent, client = await get_mcp_agent(llm)
    
    try:
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == "exit":
                print("Exiting chat...")
                break
            if user_input.lower() == "clear":
                agent.clear_memory()
                print("Memory cleared!")
                continue
            
            print("\nTutor: ", end="", flush=True)
            try:
                # Rate limit if needed
                await ddg_rate_limiter.wait_if_needed()
                
                # Process with agent
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"\nAn error occurred: {e}")
    
    finally:
        if client and client.sessions:
            await client.close_all_sessions()