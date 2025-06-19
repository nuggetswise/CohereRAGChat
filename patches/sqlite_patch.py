"""
SQLite compatibility patch for ChromaDB
This module provides a fix for environments with older SQLite versions
"""

import sys
import os

# Apply SQLite fix at import time
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If pysqlite3 is not available, try setting environment variable for ChromaDB
    try:
        os.environ['CHROMA_DB_IMPL'] = 'duckdb+parquet'
    except Exception:
        pass

# SQLite version checker function for diagnostics
def check_sqlite_version():
    """Check current SQLite version and compatibility with ChromaDB"""
    import sqlite3
    version = sqlite3.sqlite_version
    required = "3.35.0"
    
    version_parts = list(map(int, version.split('.')))
    required_parts = list(map(int, required.split('.')))
    
    meets_requirements = False
    if version_parts[0] > required_parts[0]:
        meets_requirements = True
    elif version_parts[0] == required_parts[0] and version_parts[1] > required_parts[1]:
        meets_requirements = True
    elif (version_parts[0] == required_parts[0] and 
          version_parts[1] == required_parts[1] and 
          version_parts[2] >= required_parts[2]):
        meets_requirements = True
    
    return {
        'version': version,
        'required': required,
        'meets_requirements': meets_requirements,
        'using_pysqlite3': 'pysqlite3' in sys.modules,
        'using_alternative_backend': os.environ.get('CHROMA_DB_IMPL') == 'duckdb+parquet'
    }