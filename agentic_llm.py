from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Literal

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
llm=ChatGroq(model="llama-3.1-8b-instant")
llm

@tool
def search_web(query: str) -> str:
    """A mock function to simulate web search."""
    return f"Search results for '{query}'"

@tool
def get_weather(query: str) -> list:
    """Search the web for a query"""
    tavily_search = TavilySearch(api_key=TAVILY_API_KEY,
                                 max_results=2,
                                 search_depth='advanced',
                                 max_tokens=1000)
    results = tavily_search.invoke(query)
    return results

tools = [search_web, get_weather]
llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)

# define functions to call the LLM or the tools
def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tools(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# initialize the workflow from StateGraph
workflow = StateGraph(MessagesState)

# add a node named LLM, with call_model function. This node uses an LLM to make decisions based on the input given
workflow.add_node("LLM", call_model)

# Our workflow starts with the LLM node
workflow.add_edge(START, "LLM")

# Add a tools node
workflow.add_node("tools", tool_node)

# Add a conditional edge from LLM to call_tools function. It can go tools node or end depending on the output of the LLM.
workflow.add_conditional_edges("LLM", call_tools)

# tools node sends the information back to the LLM
workflow.add_edge("tools", "LLM")

agent = workflow.compile()
from IPython.display import Image
Image(agent.get_graph().draw_mermaid_png())

for chunk in agent.stream(
    {"messages": [("user", "you are a weather expert, can you tell me whether it will rain in Trivandrum today?")]},
    stream_mode="values",):
    chunk["messages"][-1].pretty_print()
