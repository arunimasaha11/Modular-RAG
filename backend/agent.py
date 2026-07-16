import os
from typing import TypedDict, List, Literal, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, SecretStr
from config import GROQ_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY
from vector_store import get_retriever

# Tools
tavily=TavilySearch(max=3,topic="general")

@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Tavily"""
    try:
        result = tavily.invoke({"query": query})
        if isinstance(result, dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', '')
                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
            return "\n\n".join(formatted_results) if formatted_results else "No results found"
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR::{e}"


@tool
def rag_search_tool(query:str)->str:
    """Top-K chunks from KB (empty string if none)"""
    try:
        retriever_instance = get_retriever()
        docs = retriever_instance.invoke(query,k=5)
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        return f"RAG_error :: {e}"
    


# Pydantic Schemas for structured output
class RouteDecision(BaseModel): 
    route : Literal["rag","web","answer","end"]
    reply : str | None=Field(None, description="Filled only when route =='end' ")


class RagJudge(BaseModel):
    sufficient : bool = Field(..., description="True if retrieved information is sufficient to answer the user's question, False otherwise.")


# LLM instances with structured schemas

router_llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0).with_structured_output(RouteDecision)
judge_llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0).with_structured_output(RagJudge)
answer_llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

# State : Share Data Structure
class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    route : Literal["rag","web","answer","end"]
    rag : str
    web : str
    web_search_enabled : bool
    router_overrider_reason: str
    initial_router_decision: str
    
# Node : For individual function 
# Node 1 : router(decision_node)
def router_node(state:AgentState)->AgentState:
    print("Entering Router Node")
    # Extract Query
    query = next((m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),"")
    web_search_enabled = state.get("web_search_enabled",True)
    print(f"Router received web search info: {web_search_enabled}")
    
     
    system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
        "Prioritize using the **internal knowledge base (RAG)** for factual information that is likely "
        "to be contained within pre-uploaded documents or for common, well-established facts."
    )
    
    if web_search_enabled:
        system_prompt += (
            "You **CAN** use web search for queries that require very current, real-time, or broad general knowledge "
            "that is unlikely to be in a specific, static knowledge base (e.g., today's news, live data, very recent events)."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection (e.g., 'What is X?', 'How does Y work?', 'Explain Z policy')."
            "\n- 'web': For queries about current events, live data, very recent news, or broad general knowledge that requires up-to-date internet access (e.g., 'Who won the election yesterday?', 'What is the weather in London?', 'Latest news on technology')."
        )
    else:
        system_prompt += (
            "**Web search is currently DISABLED.** You **MUST NOT** choose the 'web' route."
            "If a query would normally require web search, you should attempt to answer it using RAG (if applicable) or directly from your general knowledge."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection, AND for queries that would normally go to web search but web search is disabled."
            "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        )

    system_prompt += (
        "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        "\n- 'end': For pure greetings or small-talk where no factual answer is expected (e.g., 'Hi', 'How are you?'). If choosing 'end', you MUST provide a 'reply'."
        "\n\nExample routing decisions:"
        "\n- User: 'What are the treatment of diabetes?' -> Route: 'rag' (Factual knowledge, likely in KB)."
        "\n- User: 'What is the capital of France?' -> Route: 'rag' (Common knowledge, can be in KB or answered directly if LLM knows)."
        "\n- User: 'Who won the NBA finals last night?' -> Route: 'web' (Current event, requires live data)."
        "\n- User: 'How do I submit an expense report?' -> Route: 'rag' (Internal procedure)."
        "\n- User: 'Tell me about quantum computing.' -> Route: 'rag' (Foundational knowledge can be in KB. If KB is sparse, judge will route to web if enabled)."
        "\n- User: 'Hello there!' -> Route: 'end', reply='Hello! How can I assist you today?'"
    )  
    
    prompt = ChatPromptTemplate.from_messages(
       [
         ("system", system_prompt),
         ("human", "{query}")
       ]
    )
    messages = prompt.format_messages(query=query)
    
    result = router_llm.invoke(messages)
    assert isinstance(result, RouteDecision)
    initial_router_decision = result.route
    router_overrider_reason = None
    
    # Override the router decision to go for web search 
    if not web_search_enabled and result.route == "web" :
        result.route = "rag"
        router_overrider_reason = "Web search disabled by user, Redirected to the rag"
        print(f"Router decison overridden : changed from 'web to 'rag'. ")
    
    print(f"Router Final Decision : {result.route}, reply (if 'end'): {result.reply}")
    
    out: AgentState = {
       "messages": state.get("messages", []),
       "route": result.route,
       "web_search_enabled": web_search_enabled,
    }
    
    if router_overrider_reason:
       out["router_overrider_reason"] = router_overrider_reason
       out["initial_router_decision"] = initial_router_decision
        
        
    if result.route == "end":
        out["messages"]=state.get("messages",[])+[AIMessage(content=result.reply or "Hello! How can I assist you today?")]
        
        
    print(f"Existing router_node")
    return out  

# Node 2 : RAG Lookup
def rag_node(state:AgentState)->AgentState:
    print("Entering Rag Node")
    query = next((m.content for m in reversed(state.get("messages",[])) if isinstance(m,HumanMessage)),"")
    web_search_enabled = state.get("web_search_enabled",True)
    print(f"Router received web search info: {web_search_enabled}")
    print(f"RAG Query : {query}")
    
    # Ensure query is a string to satisfy the tool.invoke type signature
    chunks = rag_search_tool.invoke(str(query))
    
    # Logic to handle chunk
    if chunks.startswith("RAG ERROR :: "):
        print(f"RAG Error :{chunks}, checking web search enabled status")
        # if RAG fails, web search is enabled
        next_route = "web" if web_search_enabled else "answer"
        return {**state,"rag":"","route":next_route}
    if chunks :
        print(f"Retrieved RAG Chunk : {chunks[:500]}....")
    else :
        print(f"No RAG Chunks retrieved")
        
    judge_messages = [
        ("system", (
            "You are a judge evaluating if the **retrieved information** is **sufficient and relevant** "
            "to fully and accurately answer the user's question. "
            "Consider if the retrieved text directly addresses the question's core and provides enough detail."
            "If the information is incomplete, vague, outdated, or doesn't directly answer the question, it's NOT sufficient."
            "If it provides a clear, direct, and comprehensive answer, it IS sufficient."
            "If no relevant information was retrieved at all (e.g., 'No results found'), it is definitely NOT sufficient."
            "\n\nRespond ONLY with a JSON object: {\"sufficient\": true/false}"
            "\n\nExample 1: Question: 'What is the capital of France?' Retrieved: 'Paris is the capital of France.' -> {\"sufficient\": true}"
            "\nExample 2: Question: 'What are the symptoms of diabetes?' Retrieved: 'Diabetes is a chronic condition.' -> {\"sufficient\": false} (Doesn't answer symptoms)"
            "\nExample 3: Question: 'How to fix error X in software Y?' Retrieved: 'No relevant information found.' -> {\"sufficient\": false}"
        )),
        ("user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient to answer the question?")
    ]
    
    verdict = judge_llm.invoke(judge_messages)
    assert isinstance(verdict, RagJudge)
    print(f"RAG Judge Verdict : {verdict.sufficient}")
    print("Exiting rag_node")
    
    # Decide next route based on sufficiency and web_search info
    if verdict.sufficient :
        next_route="answer"
    else:
        next_route="web" if web_search_enabled else "answer"
        print(f"RAG not sufficient. Web search enabled : {web_search_enabled}. Next Route : {next_route}")
        
    return {
        **state,
        "rag" : chunks,
        "route" : next_route,
        "web_search_enabled" : web_search_enabled 
    }
    
# Node 3 : Web Search

def web_node(state:AgentState)->AgentState:
    print("Entering Web Node")
    query = next((m.content for m in reversed(state.get("messages",[])) if isinstance(m,HumanMessage)),"")
    web_search_enabled = state.get("web_search_enabled",True)
    if not web_search_enabled :
        print("Web Search node entered but search is disabled.")
        return {**state,"web":"Web search was disabled by user","route":"answer"}
    print(f"Web Search Query : {query}")
    snippets = web_search_tool.invoke({"query": query})
    
    if snippets.startswith("WEB ERROR :: "):
        print(f"Web Error : {snippets}. Predicting to answer with limited info")
        return {
            **state,
            "web" : "",
            "route" : "answer"
        }
    
    print(f"Web Snippets Retrieved : {snippets[:500]}")
    print("Exiting web_node")
    return {
        **state,
        "web" : snippets,
        "route" : "answer"
    }
    
# Node 4 : Final Node
def answer_node(state:AgentState)->AgentState:
    print("Entering answer_node")    
    user_query = next((m.content for m in reversed(state.get("messages",[])) if isinstance(m,HumanMessage)),"")
    
    ctx_parts = []
    rag = state.get("rag")
    if rag:
      ctx_parts.append("Knowledge Base Information:\n" + rag)
    web = state.get("web")
    if web and not web.startswith("Web search was disabled"):
        ctx_parts.append("Web Search Results:\n" + web)
            
    context = "\n\n".join(ctx_parts)
    if not context.strip():
        context = "No external Context was available for this query. Try to answer based on General Knowledge."
        
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human","""Please answer the user's question using the provided context. If the context is empty or irrelevant, try to answer based on your general knowledge.
            Question : {user_query}
            Context : {context}
            Provide a helpful, accurate, and concise response based on the available information.""",
        )
    ]
    )

    messages = prompt.format_messages(
        user_query=user_query,
        context=context
    )
    response = answer_llm.invoke(messages)
    answer = response.content
    print(f"Final Answer : {answer[:500]}")
    print("Exiting answer_node")
    return {
        **state,
        "messages": state.get("messages",[]) +[AIMessage(content=answer)]
    }
    
# --- Routing helpers ---
def from_router(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
    return st.get("route", "answer")

def after_rag(st: AgentState) -> Literal["answer", "web"]:
    # Ensure we only return one of the allowed literals
    route = st.get("route", "answer")
    if route in ("answer", "web"):
        return route
    return "answer"

def after_web(_) -> Literal["answer"]:
    return "answer" 

    
# Building Graph
def build_agent():
    """Build and Compile LangGraph Agents."""
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("rag_lookup", rag_node)
    graph.add_node("web_search", web_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("router")
    
    graph.add_conditional_edges(
        "router",
        from_router,
        {
            "rag": "rag_lookup",
            "web": "web_search",
            "answer": "answer",
            "end": END
        }
    )
    
    graph.add_conditional_edges(
        "rag_lookup",
        after_rag,
        {
            "answer": "answer",
            "web": "web_search"
        }
    )
    
    graph.add_conditional_edges(
        "web_search",
        after_web,
        {
            "answer" : "answer"
        }
    )
    
    # graph.add_edge("web_search", "answer")
    graph.add_edge("answer", END)

    agent = graph.compile(checkpointer=MemorySaver())
    return agent

rag_agent = build_agent()