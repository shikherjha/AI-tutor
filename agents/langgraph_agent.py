"""
LangGraph agent implementation for Axon AI Tutor
"""
from typing import Dict, Any, List, TypedDict, Annotated
import operator
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

# Define state for our LangGraph agent
class AgentState(TypedDict):
    """State for the LangGraph agent"""
    question: str
    context: str
    file_context: str
    search_results: List[Dict[str, str]]
    need_search: bool
    final_answer: str

def create_langgraph_agent(llm: BaseChatModel, tavily_api_key: str):
    """
    Create a LangGraph-based educational agent.
    
    Args:
        llm: Language model to use
        tavily_api_key: API key for Tavily search
        
    Returns:
        A compiled LangGraph agent
    """
    # Create Tavily search tool
    tavily_tool = TavilySearchResults(
        api_key=tavily_api_key,
        max_results=5,
        include_domains=["edu", "org", "gov"]
    )
    
    # Define node functions
    def determine_search_need(state: AgentState) -> str:
        """Determine if search is needed"""
        question = state["question"]
        
        # Prompt to determine if search is needed
        prompt = ChatPromptTemplate.from_template(
            """Given the student's question, determine if external search would be beneficial.
            
            Question: {question}
            
            Return a JSON with the following format:
            {{
                "need_search": true or false,
                "reasoning": "brief explanation of your decision"
            }}
            """
        )
        
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"question": question})
        
        state["need_search"] = result.get("need_search", False)
        return "search" if state["need_search"] else "answer"
    
    async def search_for_information(state: AgentState) -> AgentState:
        """Search for information"""
        # Use Tavily to search
        search_results = await tavily_tool.ainvoke(state["question"])
        
        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "url": result["url"],
                "title": result["title"],
                "content": result["content"]
            })
            
        state["search_results"] = formatted_results
        
        # Format as context
        search_context = "\n".join([
            f"Source: {result['url']}\nTitle: {result['title']}\nContent: {result['content']}"
            for result in search_results
        ])
        
        state["context"] = search_context
        return state
    
    def generate_answer(state: AgentState) -> AgentState:
        """Generate educational answer"""
        # Combine all context
        context = ""
        if state["file_context"]:
            context += f"Context from uploaded files:\n{state['file_context']}\n\n"
        if state.get("context"):
            context += f"Search results:\n{state['context']}"
            
        # Educational prompt
        prompt = ChatPromptTemplate.from_template(
            """You are an AI tutor specializing in education.
            
            The student has asked: {question}
            
            {context}
            
            Provide a clear, accurate, and educational response. Include examples where helpful.
            Break down complex concepts and explain your reasoning.
            
            If mathematical or scientific notation would help, use proper formatting.
            If code examples would help, provide properly commented code.
            
            If you used search results, cite your sources at the end.
            """
        )
        
        # Generate answer
        chain = prompt | llm 
        answer = chain.invoke({
            "question": state["question"],
            "context": context
        })
        
        state["final_answer"] = answer
        return state
    
    # Define the LangGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("determine_search", determine_search_need)
    workflow.add_node("search", search_for_information)
    workflow.add_node("answer", generate_answer)
    
    # Add edges
    workflow.add_edge("determine_search", "search", condition=lambda x: x == "search")
    workflow.add_edge("determine_search", "answer", condition=lambda x: x == "answer")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    
    # Compile the graph
    return workflow.compile()

# Example usage:
# langgraph_agent = create_langgraph_agent(llm, TAVILY_API_KEY) 
# result = await langgraph_agent.ainvoke({
#     "question": "What is photosynthesis?",
#     "context": "",
#     "file_context": "",
#     "search_results": [],
#     "need_search": False,
#     "final_answer": ""
# })
# answer = result["final_answer"]