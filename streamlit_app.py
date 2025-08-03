import streamlit as st
import requests
import json
import time
import re

API_URL = "http://localhost:8000"

def call_api(query):
    """Call the API with the user query"""
    try:
        payload = {"text": query}
        response = requests.post(
            f"{API_URL}/chat/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("API connection error. Please ensure the FastAPI backend is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/chat/health")
        return response.status_code == 200
    except:
        return False

def format_thinking(messages):
    """Format the thinking process with icons and clear sections"""
    thinking = ""
    current_agent = "supervisor"
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system" and "transferred to" in content:
            agent = content.split("to ")[ -1].strip()
            thinking += f"🔄 **Routing to {agent}**\n\n"
            current_agent = agent
            continue
        elif role == "system" and "transferred back" in content:
            thinking += f"🔙 **Returning to Supervisor**\n\n"
            current_agent = "supervisor"
            continue
        
        if role == "human":
            thinking += f"👤 **Human Query**: {content}\n\n"
            continue
            
        if role == "tool":
            if "Weather in" in content:
                tool_name = "Weather Tool"
                thinking += f"🌤️ **{tool_name}**: {content}\n\n"
            elif "answer='" in content:
                tool_name = "Document Tool"
                # Extract the answer from the format answer='...'
                answer = content.split("answer='", 1)[1].rsplit("'", 1)[0]
                thinking += f"📄 **{tool_name}**: {answer}\n\n"
            else:
                thinking += f"🔧 **Tool Call**: {content}\n\n"
            continue
            
        if role == "ai":
            thinking += f"🤖 **AI ({current_agent})**: {content}\n\n"
            continue
            
        thinking += f"ℹ️ **System**: {content}\n\n"
    
    return thinking

def extract_professional_response(messages):
    """Extract the final professional response from the conversation"""
    tool_output = None
    tool_name = None
    
    # First look for tool outputs
    for msg in messages:
        if msg["role"] == "tool":
            content = msg["content"]
            if "Weather in" in content:
                tool_name = "Weather Tool"
                tool_output = content
                break
            elif "answer='" in content:
                tool_name = "Document Tool"
                # Extract the answer from the format answer='...'
                tool_output = content.split("answer='", 1)[1].rsplit("'", 1)[0]
                break
    
    # Then look for the final AI response
    ai_response = None
    for msg in reversed(messages):
        if msg["role"] == "ai":
            content = msg["content"]
            # Skip routing messages and generic responses
            if any(phrase in content.lower() for phrase in [
                "transferring back", 
                "conversation completed", 
                "please enter your next query",
                "should be routed",
                "i'll route this"
            ]):
                continue
            ai_response = content
            break
    
    # Determine what to return
    if tool_output and not ai_response:
        return f"{tool_output}"
    elif ai_response and not tool_output:
        return ai_response
    elif tool_output and ai_response:
        return f"{ai_response}\n\n_Retrieved using {tool_name}_"
    else:
        return "I'm sorry, I couldn't find specific information about that in the available documents."

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
    st.title("🤖 AI Assistant")
    st.write("Ask questions about the weather or about Abhinandan's resume")
    
    # Check if API is running
    api_running = check_api_health()
    if not api_running:
        st.error("⚠️ API is not running. Please start the API server first.")
        st.info("Run the API with: `cd src && uvicorn main:app --reload`")
        return
    else:
        st.success("✅ Connected to API")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"].lower()):
            st.write(message["content"])
    
    # Get user input
    user_query = st.chat_input("Ask a question...")
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        # Process the query
        with st.spinner("Thinking..."):
            result = call_api(user_query)
            
            if result:
                # Format the thinking process
                thinking = format_thinking(result["messages"])
                final_response = extract_professional_response(result["messages"])
                
                # Display thinking process
                with st.expander("See thinking process", expanded=True):
                    st.markdown(thinking)
                    st.write(f"Processing time: {result['processing_time']:.2f} seconds")
                    with st.expander("Raw API Response"):
                        st.json(result)
                
                # Display final response
                with st.chat_message("assistant"):
                    st.write(final_response)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main()