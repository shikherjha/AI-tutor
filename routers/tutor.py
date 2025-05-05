"""
Tutor endpoints for Axon AI Tutor
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends

from models.api_models import TutorQuery, TutorResponse
from core.llm import get_llm
from core.memory import get_or_create_conversation
from core.rate_limiter import tavily_rate_limiter, ddg_rate_limiter
from agents.mcp_agent import get_mcp_agent, extract_sources_from_agent_response
from agents.langgraph_agent import create_langgraph_agent
from processors.document_processor import DocumentProcessor
from processors.translation import translate_text, detect_language
from core.state import conversations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

from config.settings import TAVILY_API_KEY, DEFAULT_LANGUAGE

# Create router
router = APIRouter()

# Initialize document processor
doc_processor = DocumentProcessor()

def _needs_external_tools(question: str) -> bool:
    """Determine if the question likely needs external tools."""
    external_indicators = [
        "latest", "recent", "news", "current", 
        "search", "find", "lookup", "research",
        "what is the current", "how many", "statistics",
        "recommend", "where can i", "what are the best"
    ]
    return any(indicator in question.lower() for indicator in external_indicators)

@router.post("/ask", response_model=TutorResponse)
async def ask_tutor(request: TutorQuery):
    """API endpoint for asking questions to the AI tutor."""
    try:
        # Get or create conversation
        conversation_id, conversation_data = get_or_create_conversation(
            conversations, request.conversation_id
        )
        
        # Initialize LLM
        llm = get_llm()
        
        # Check if we need to translate the question
        source_language = None
        if request.language != "english" and request.language != "en":
            # Detect the language if not specified
            if request.language == "auto":
                source_language = await detect_language(request.question)
            else:
                source_language = request.language
                
            # Translate to English if needed
            if source_language != "en" and source_language != "english":
                original_question = request.question
                request.question = await translate_text(
                    request.question, 
                    source_language=source_language,
                    target_language="en"
                )
        
        # Check for file context
        file_context = doc_processor.find_relevant_context(conversation_data, request.question)
        
        # Determine if we need external tools
        use_tools = _needs_external_tools(request.question)
        
        # Process the question and generate response
        if use_tools:
            if request.use_langgraph:
                # Use LangGraph agent for more sophisticated reasoning
                langgraph_agent = create_langgraph_agent(llm, TAVILY_API_KEY)
                
                # Prepare initial state
                initial_state = {
                    "question": request.question,
                    "context": "",
                    "file_context": file_context or "",
                    "search_results": [],
                    "need_search": False,
                    "final_answer": ""
                }
                
                # Execute the graph
                result = await langgraph_agent.ainvoke(initial_state)
                response = result["final_answer"]
                
                # Extract sources
                sources = []
                for result in result.get("search_results", []):
                    if "url" in result:
                        sources.append(result["url"])
                
            elif request.use_tavily and TAVILY_API_KEY:
                # Use Tavily for academic/reliable search
                await tavily_rate_limiter.wait_if_needed()
                
                # Initialize Tavily search tool
                tavily_tool = TavilySearchResults(
                    api_key=TAVILY_API_KEY,
                    max_results=5,
                    include_domains=["edu", "org", "gov"]
                )
                
                # Perform search
                search_results = await tavily_tool.ainvoke(request.question)
                
                # Format search results for context
                search_context = "\n".join([
                    f"Source: {result['url']}\nTitle: {result['title']}\nContent: {result['content']}"
                    for result in search_results
                ])
                
                # Create combined context
                context = ""
                if file_context:
                    context += f"Context from uploaded files:\n{file_context}\n\n"
                context += f"Search results:\n{search_context}"
                
                # Create educational prompt with search context
                template = """You are an AI tutor specializing in education.
                
                The student has asked: {question}
                
                Here is relevant information to help you answer:
                {context}
                
                Provide a clear, accurate, and educational response. Include examples where helpful.
                Break down complex concepts and explain your reasoning.
                
                If mathematical or scientific notation would help, use proper formatting.
                If code examples would help, provide properly commented code.
                
                At the end of your response, include sources that you used from the search results.
                """
                
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | llm | StrOutputParser()
                
                # Run chain with context
                response = await chain.ainvoke({
                    "question": request.question,
                    "context": context
                })
                
                # Extract sources
                sources = [result['url'] for result in search_results]
                
            elif request.use_mcp:
                # Use MCP agent with web search
                await ddg_rate_limiter.wait_if_needed()
                
                # Initialize MCP agent if not already created
                if not conversation_data.get("agent"):
                    agent, client = await get_mcp_agent(llm)
                    conversation_data["agent"] = agent
                    conversation_data["client"] = client
                
                # Add file context if available
                question_with_context = request.question
                if file_context:
                    question_with_context = f"""
                    I have the following information from my documents:
                    {file_context}
                    
                    Based on this and your knowledge, please answer: {request.question}
                    """
                
                # Run the agent
                agent = conversation_data["agent"]
                response = await agent.run(question_with_context)
                sources = extract_sources_from_agent_response(response)
            else:
                # Fallback to direct LLM if no tool usage specified
                response = await _direct_llm_response(llm, request.question, file_context)
                sources = []
        else:
            # Use direct LLM chain for educational content
            response = await _direct_llm_response(llm, request.question, file_context)
            sources = []
        
        # Translate response back if needed
        if source_language and source_language != "en" and source_language != "english":
            response = await translate_text(
                response,
                source_language="en",
                target_language=source_language
            )
        
        # Update conversation memory
        conversation_data["memory"].chat_memory.add_user_message(request.question)
        conversation_data["memory"].chat_memory.add_ai_message(response)
        
        return TutorResponse(
            answer=response,
            conversation_id=conversation_id,
            sources=sources,
            language=request.language
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def _direct_llm_response(llm, question: str, file_context: Optional[str]) -> str:
    """Generate direct LLM response without using tools"""
    template = """You are an AI tutor specializing in education.
    
    The student has asked: {question}
    
    {context}
    
    Provide a clear, accurate, and educational response. Include examples where helpful.
    Break down complex concepts and explain your reasoning.
    
    If mathematical or scientific notation would help, use proper formatting.
    If code examples would help, provide properly commented code.
    """
    
    context = ""
    if file_context:
        context = f"Here is relevant information from uploaded documents:\n{file_context}"
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    return await chain.ainvoke({
        "question": question,
        "context": context
    })