"""
LangChain Agent with Streamlit - Simple UI
"""

import streamlit as st
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AI Agent Chat",
    page_icon="🤖",
    layout="centered"
)

# =============================================================================
# TOOLS
# =============================================================================

@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            weather_desc = current['weatherDesc'][0]['value']
            temp_c = current['temp_C']
            feels_like = current['FeelsLikeC']
            humidity = current['humidity']
            return f"Weather in {city}: {weather_desc}, Temperature: {temp_c}°C (feels like {feels_like}°C), Humidity: {humidity}%"
        else:
            return f"Could not fetch weather for {city}"
    except Exception as e:
        return f"Error getting weather: {str(e)}"

# =============================================================================
# AGENT
# =============================================================================

@st.cache_resource
def create_agent():
    """Initialize the agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    tavily_tool = TavilySearchResults(max_results=3, search_depth="basic")
    tools = [get_current_datetime, get_weather, tavily_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant with access to datetime, weather, and web search tools."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_executor" not in st.session_state:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("⚠️ Missing API keys! Add OPENAI_API_KEY and TAVILY_API_KEY to your .env file")
        st.stop()
    st.session_state.agent_executor = create_agent()

# =============================================================================
# UI
# =============================================================================

st.title("🤖 AI Agent Chat")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                formatted_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        formatted_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        formatted_history.append(AIMessage(content=msg["content"]))
                
                response = st.session_state.agent_executor.invoke({
                    "input": prompt,
                    "chat_history": formatted_history
                })
                
                assistant_response = response['output']
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})