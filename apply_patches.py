#!/usr/bin/env python
"""

Script to apply compatibility patches to the installed packages.
This should be run before starting the Streamlit app.
"""
import os
import sys
import importlib.util
from pathlib import Path

def patch_sqlite_for_chromadb():
    """
    Universal SQLite fix for ChromaDB compatibility across all environments.
    This is especially important for production environments where the system
    SQLite version might be older than 3.35.0 which ChromaDB requires.
    """
    try:
        # Try to import pysqlite3 and replace the system sqlite3
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("✅ SQLite patched successfully for ChromaDB compatibility")
        return True
    except ImportError:
        print("⚠️ pysqlite3 not available - attempting fallback options")
        
        # Try to set ChromaDB to use alternative backend
        try:
            os.environ['CHROMA_DB_IMPL'] = 'duckdb+parquet'
            print("ℹ️ Set ChromaDB to use DuckDB backend instead of SQLite")
        except Exception:
            pass
            
        return False

def patch_langchain_cohere():
    """
    Patches the langchain-cohere library to use our compatibility classes.
    """
    try:
        # Find the location of the langchain_cohere package
        import langchain_cohere
        package_path = Path(langchain_cohere.__file__).parent
        cohere_agent_path = package_path / "cohere_agent.py"
        
        if not cohere_agent_path.exists():
            print(f"Could not find {cohere_agent_path}")
            return False
        
        # Read the original file content
        with open(cohere_agent_path, 'r') as file:
            content = file.read()
        
        # Check if we need to patch
        if "from cohere.types import (" in content and "ToolResult," in content:
            # Replace the import line with our patch
            patched_content = content.replace(
                "from cohere.types import (",
                "# Modified import to use compatibility patch\n"
                "try:\n"
                "    from cohere.types import (\n"
                "        Tool,\n"
                "        ToolCall,\n"
                "        ToolParameterDefinitionsValue,\n"
                "    )\n"
                "    # Import ToolResult from our patch\n"
                "    from patches.cohere_types_patch import ToolResult\n"
                "except ImportError:\n"
                "    # Fallback to original imports\n"
                "    from cohere.types import ("
            )
            
            # Write the patched content
            with open(cohere_agent_path, 'w') as file:
                file.write(patched_content)
            
            print(f"✅ Successfully patched {cohere_agent_path}")
            return True
        else:
            print("File already patched or has unexpected format")
            return False
    except Exception as e:
        print(f"Error patching langchain-cohere: {e}")
        return False

if __name__ == "__main__":
    # IMPORTANT: Apply SQLite patch first, before any other imports
    patch_sqlite_for_chromadb()
    
    # Add the current directory to Python path to make our patches available
    current_dir = Path(__file__).parent.absolute()
    sys.path.insert(0, str(current_dir))
    
    # Apply patches
    success = patch_langchain_cohere()
    
    if success:
        print("All patches applied successfully!")
    else:
        print("Some patches failed. Check the errors above.")
