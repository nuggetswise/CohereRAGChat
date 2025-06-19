"""
RAG Chat Application - Standalone Version
Chat with your documents using Cohere's powerful LLMs
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import modules
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Apply SQLite patch first thing
from apply_patches import patch_sqlite_for_chromadb
patch_sqlite_for_chromadb()

# Import dependency check
try:
    from dependency_check import run_dependency_check
    run_dependency_check()
except ImportError:
    st.warning("Dependency check module not found. Proceeding without dependency verification.")

# Page configuration
st.set_page_config(
    page_title="RAG Chat",
    page_icon="💬",
    layout="wide"
)

# Import the RAG chat page
try:
    sys.path.insert(0, str(Path(__file__).parent / "pages"))
    from pages.5_RAG_Chat import app as rag_chat_app
except ImportError as e:
    st.error(f"Error importing RAG Chat: {str(e)}")
    rag_chat_app = None

def main():
    st.title("💬 RAG Chat")
    st.caption("Chat with your documents powered by Cohere")
    
    # Get API key from secrets or environment
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key and hasattr(st, "secrets"):
        if "COHERE_API_KEY" in st.secrets:
            cohere_key = st.secrets["COHERE_API_KEY"]
        elif "cohere" in st.secrets and "COHERE_API_KEY" in st.secrets.cohere:
            cohere_key = st.secrets.cohere["COHERE_API_KEY"]
    
    # Show API key input if needed
    if not cohere_key:
        with st.expander("⚙️ API Key Settings"):
            cohere_key = st.text_input("Enter your Cohere API key:", type="password")
            if cohere_key:
                st.success("API key set for this session")
    
    # Run the RAG chat app
    if cohere_key and rag_chat_app:
        rag_chat_app()
    elif not cohere_key:
        st.info("Please provide a Cohere API key to start chatting.")
    else:
        st.error("RAG Chat application could not be loaded.")

if __name__ == "__main__":
    main()