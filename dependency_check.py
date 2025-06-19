"""
Streamlit app dependency checker
Runs automatically when the streamlit app launches to verify critical dependencies
"""
import streamlit as st
import os
import sys
import importlib
from typing import Dict, List, Tuple

# CRITICAL: Apply SQLite patch at the very beginning before any imports
try:
    # First try to import and use pysqlite3
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ SQLite patch applied in dependency_check.py")
except ImportError:
    # If pysqlite3 is not available, try setting environment variable for ChromaDB
    try:
        os.environ['CHROMA_DB_IMPL'] = 'duckdb+parquet'
        print("ℹ️ Set ChromaDB to use DuckDB backend instead of SQLite")
    except Exception:
        print("⚠️ Failed to apply SQLite patches - this may cause issues")

# List of critical dependencies with versions
CRITICAL_DEPS = {
    "chromadb": ">=0.4.18",
    "cohere": ">=5.0.0",
    "pandas": ">=2.0.0",
    "crewai": ">=0.28.0"
}

def check_sqlite_version() -> Tuple[bool, str]:
    """Check if SQLite version meets ChromaDB requirements"""
    try:
        import sqlite3
        version = sqlite3.sqlite_version
        version_info = tuple(map(int, version.split('.')))
        required_version = (3, 35, 0)
        
        # Check if we're using pysqlite3
        using_pysqlite3 = 'pysqlite3' in sys.modules
        
        if version_info >= required_version:
            return True, f"SQLite version {version} meets ChromaDB requirements"
        elif using_pysqlite3:
            # If we're using pysqlite3 but version check shows lower version,
            # it means the patch wasn't fully applied
            return False, f"SQLite version {version} reported, but pysqlite3 patch might not be working correctly"
        else:
            return False, f"SQLite version {version} is too old (ChromaDB requires >= 3.35.0)"
    except Exception as e:
        return False, f"Failed to check SQLite version: {str(e)}"

def check_dependency(name: str) -> bool:
    """Check if a dependency is installed"""
    try:
        # Special handling for ChromaDB which can raise RuntimeError
        if name == "chromadb":
            try:
                # Check SQLite version first
                sqlite_ok, sqlite_msg = check_sqlite_version()
                if not sqlite_ok:
                    st.error(f"""
                    ## ❌ SQLite Version Error
                    
                    {sqlite_msg}
                    
                    ### This is a common issue with ChromaDB
                    
                    ChromaDB requires SQLite 3.35.0 or higher, but your system has an older version.
                    
                    ### How to Fix:
                    
                    1. **Option 1**: Install pysqlite3 package (recommended):
                    ```
                    pip install pysqlite3-binary
                    ```
                    
                    2. **Option 2**: Upgrade your system's SQLite (may require admin rights):
                    ```
                    # On Ubuntu/Debian:
                    sudo apt-get update && sudo apt-get install sqlite3
                    
                    # On macOS with Homebrew:
                    brew install sqlite
                    ```
                    
                    For more details, visit: https://docs.trychroma.com/troubleshooting#sqlite
                    """)
                    return False
                
                importlib.import_module(name)
                return True
            except RuntimeError as e:
                # Check if it's the SQLite error
                if "unsupported version of sqlite3" in str(e):
                    st.error(f"""
                    ## ❌ SQLite Version Error
                    
                    ChromaDB requires SQLite 3.35.0 or higher, but your system has an older version.
                    
                    Error importing crewai: {str(e)}
                    
                    ### How to Fix:
                    
                    1. **Option 1**: Install pysqlite3 package (recommended):
                    ```
                    pip install pysqlite3-binary
                    ```
                    
                    2. **Option 2**: Upgrade your system's SQLite (may require admin rights):
                    ```
                    # On Ubuntu/Debian:
                    sudo apt-get update && sudo apt-get install sqlite3
                    
                    # On macOS with Homebrew:
                    brew install sqlite
                    ```
                    
                    For more details, visit: https://docs.trychroma.com/troubleshooting#sqlite
                    """)
                else:
                    # This is likely a ChromaDB dependency issue
                    st.error(f"""
                    ## ❌ ChromaDB Dependency Error
                    
                    ChromaDB is installed but raised an error during import: 
                    
                    ```
                    {str(e)}
                    ```
                    
                    ### Common Solutions:
                    
                    1. Install required dependencies:
                    ```
                    pip install chromadb>=0.4.22 onnxruntime>=1.16.0 tokenizers>=0.13.3
                    ```
                    
                    2. If on Apple Silicon Mac, try:
                    ```
                    pip install --upgrade onnxruntime
                    ```
                    
                    3. If you're on a resource-constrained environment:
                    ```
                    ONNX_PROVIDER=CPU pip install --upgrade chromadb
                    ```
                    """)
                return False
        else:
            # Normal import for other packages
            importlib.import_module(name)
            return True
    except ImportError:
        return False
    except Exception as e:
        # General exception handler for other unexpected errors
        st.error(f"Error importing {name}: {str(e)}")
        return False

def verify_dependencies() -> Tuple[bool, List[str]]:
    """Verify all critical dependencies are installed"""
    missing = []
    for dep in CRITICAL_DEPS:
        if not check_dependency(dep):
            missing.append(dep)
    return len(missing) == 0, missing

def verify_api_keys() -> Tuple[bool, List[str]]:
    """Verify that necessary API keys are set"""
    required_keys = {
        "OPENAI_API_KEY": ["openai", "OPENAI_API_KEY"],
        "COHERE_API_KEY": ["cohere", "COHERE_API_KEY"]
    }
    missing = []
    
    # Check for keys in various locations
    for key, nested_path in required_keys.items():
        # Check in environment variables first
        if os.environ.get(key):
            continue
            
        # Check in Streamlit secrets (both flat and nested structure)
        if hasattr(st, "secrets"):
            # Check direct access first
            if key in st.secrets:
                continue
                
            # Check nested section
            section = nested_path[0]
            section_key = nested_path[1]
            
            if section in st.secrets and section_key in st.secrets[section]:
                continue
                
        # If we get here, the key wasn't found anywhere
        missing.append(key)
            
    return len(missing) == 0, missing

def show_dependency_warning(missing_deps: List[str], missing_keys: List[str]):
    """Show a warning about missing dependencies or API keys"""
    st.error("⚠️ Critical dependencies or API keys are missing!")
    
    if missing_deps:
        st.markdown("### Missing Dependencies")
        st.markdown("The following critical dependencies are missing:")
        for dep in missing_deps:
            st.markdown(f"- `{dep}` {CRITICAL_DEPS[dep]}")
        
        st.markdown("### How to Fix")
        st.code(f"pip install {' '.join([f'{dep}{CRITICAL_DEPS[dep]}' for dep in missing_deps])}")
        
    if missing_keys:
        st.markdown("### Missing API Keys")
        st.markdown("The following API keys are missing:")
        for key in missing_keys:
            st.markdown(f"- `{key}`")
        
        st.markdown("### How to Set API Keys")
        st.markdown("""
        You can set API keys in one of these ways:
        1. Add them to your environment variables
        2. Add them to `.streamlit/secrets.toml` file using either format:
        """)
        
        # Flat format example
        flat_example = "\n".join([f"{key} = 'your-{key.lower()}-here'" for key in missing_keys])
        
        # Nested format example (matches your current structure)
        nested_example = ""
        if "OPENAI_API_KEY" in missing_keys:
            nested_example += "[openai]\nOPENAI_API_KEY = 'your-openai-api-key-here'\n\n"
        if "COHERE_API_KEY" in missing_keys:
            nested_example += "[cohere]\nCOHERE_API_KEY = 'your-cohere-api-key-here'"
        
        st.code("# Flat structure (either works):\n" + flat_example, language="toml")
        st.code("# Nested structure (recommended):\n" + nested_example, language="toml")
    
    st.markdown("### Run Dependency Check Script")
    st.markdown("For a more detailed dependency check, run:")
    st.code("python test_chromadb_import.py")
    
    st.stop()

def run_dependency_check():
    """Run dependency check before loading the main app"""
    deps_ok, missing_deps = verify_dependencies()
    keys_ok, missing_keys = verify_api_keys()
    
    if not (deps_ok and keys_ok):
        show_dependency_warning(missing_deps, missing_keys)