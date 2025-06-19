"""
Centralized configuration and prompt templates for the RAG Chat application.
This module provides consistent settings across the application.
"""

class RAGConfiguration:
    """Configuration for the RAG system including models, parameters, and thresholds."""
    
    # Embedding and retrieval settings
    EMBEDDING_MODEL = "embed-english-v3.0"
    RERANK_MODEL = "rerank-english-v3.0"
    TOP_K_RESULTS = 3
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Reranking settings
    RERANK_MINIMUM_RELEVANCE_THRESHOLD = 0.15  # Minimum score to consider a result relevant
    RERANK_RELEVANCE_THRESHOLD = 0.70  # Threshold for good relevance
    
    # Web search settings
    WEB_SEARCH_RESULTS = 5
    WEB_SEARCH_THRESHOLD = 0.5
    
    # Generation settings
    GENERATION_MODEL_PRIMARY = "command-r-plus"
    GENERATION_MODEL_OPENAI = "gpt-3.5-turbo"
    GENERATION_TEMPERATURE = 0.3
    GENERATION_MAX_TOKENS = 800
    GENERATION_FALLBACK_MAX_TOKENS = 500
    
    # File processing settings
    MAX_FILE_SIZE_MB = 10
    SUPPORTED_FILE_TYPES = {
        "pdf": {"icon": "📄", "description": "PDF Document"},
        "txt": {"icon": "📝", "description": "Text File"},
        "csv": {"icon": "📊", "description": "CSV Data"},
        "docx": {"icon": "📘", "description": "Word Document"},
        "jpg": {"icon": "🖼️", "description": "JPEG Image"},
        "jpeg": {"icon": "🖼️", "description": "JPEG Image"},
        "png": {"icon": "🖼️", "description": "PNG Image"}
    }
    
    @staticmethod
    def should_apply_reranking(results, client):
        """Determine if reranking should be applied."""
        # Apply reranking if we have multiple results and Cohere client
        return len(results) > 1 and client is not None
    
    @staticmethod
    def process_rerank_results(rerank_response, original_docs, min_threshold=0.15):
        """Process Cohere rerank response and return reranked documents with scores."""
        reranked_results = []
        rerank_scores = []
        low_relevance = False
        
        # Process rerank results and annotate documents with scores
        for result in rerank_response.results:
            if result.relevance_score >= min_threshold:
                doc = original_docs[result.index]
                doc.metadata["rerank_score"] = result.relevance_score
                reranked_results.append(doc)
                rerank_scores.append(result.relevance_score)
        
        # Check if best result is below the good relevance threshold
        if reranked_results and reranked_results[0].metadata["rerank_score"] < RAGConfiguration.RERANK_RELEVANCE_THRESHOLD:
            low_relevance = True
        
        return reranked_results, rerank_scores, low_relevance
    
    @staticmethod
    def get_web_search_config():
        """Get web search configuration parameters."""
        return {
            "chunk_min_length": 50,
            "max_results_to_process": 1000
        }
    
    @staticmethod
    def get_web_search_queries(query):
        """Generate search queries for web search with different strategies."""
        return [
            query,  # Original query
            f"latest information about {query}",  # Focus on recency
            f"{query} facts data research",  # Focus on factual information
            f"{query} explained in detail",  # Focus on explanations
        ]
    
    @staticmethod
    def get_cross_source_prompt(query, db_context="", upload_context="", web_context=""):
        """Generate prompt for cross-source RAG response generation."""
        contexts = []
        if db_context:
            contexts.append(f"COMPENSATION DATABASE INFORMATION:\n{db_context}")
        if upload_context:
            contexts.append(f"UPLOADED DOCUMENT INFORMATION:\n{upload_context}")
        if web_context:
            contexts.append(f"WEB SEARCH INFORMATION:\n{web_context}")
        
        all_contexts = "\n\n".join(contexts)
        
        return f"""
# USER QUERY
{query}

# RELEVANT INFORMATION
{all_contexts}

# INSTRUCTIONS
1. Answer the user's query thoroughly and accurately based on the provided information
2. Cite specific data points from the information provided
3. If different sources provide conflicting information, acknowledge this and explain the differences
4. If the information is incomplete or doesn't fully answer the query, acknowledge the limitations
5. Format your response with clear headings, bullet points, and structure for readability
6. Use markdown formatting for better readability

# RESPONSE
"""
    
    @staticmethod
    def get_simple_rag_evaluation_prompt(query, contexts_str, response, top_rerank_score=None):
        """Generate prompt for evaluating RAG response quality."""
        rerank_info = f"Top rerank relevance score: {top_rerank_score:.2f}" if top_rerank_score is not None else "Rerank score not available"
        
        return f"""
# RAG EVALUATION TASK
Evaluate the quality of an AI response in a RAG (Retrieval-Augmented Generation) system.

## User Query
{query}

## Retrieved Contexts
{contexts_str}

## AI Response
{response}

## Relevance Information
{rerank_info}

# EVALUATION INSTRUCTIONS
1. Evaluate the response on the following dimensions:
   - Relevance: Does the response address the user's query?
   - Factual Accuracy: Does the response contain factual errors compared to the retrieved contexts?
   - Groundedness: Is the response well-grounded in the retrieved contexts without hallucinations?

2. For each dimension, provide:
   - A score from 1-10 (where 10 is best)
   - Brief feedback explaining the score

3. Calculate an overall score from 1-10 based on all dimensions

4. List 2-3 specific strengths of the response
5. List 2-3 specific areas for improvement

# OUTPUT FORMAT
Provide your evaluation in the following JSON format:
```json
{{
  "overall_score": 7.5,
  "relevance": {{
    "score": 8,
    "feedback": "The response directly addresses the main aspects of the query..."
  }},
  "factual_accuracy": {{
    "score": 7,
    "feedback": "The response correctly states most facts from the context but..."
  }},
  "groundedness": {{
    "score": 8,
    "feedback": "The response stays well-grounded in the provided context..."
  }},
  "strengths": [
    "Clearly addresses the main question",
    "Provides specific data points from the context"
  ],
  "areas_for_improvement": [
    "Could acknowledge uncertainty about X",
    "Could provide more comprehensive information about Y"
  ]
}}
```
"""

class RAGAnalysisSteps:
    """Predefined analysis steps and messages for the RAG pipeline."""
    
    @staticmethod
    def response_generation():
        """Return step and message for response generation phase."""
        return "Response", "🧠 Generating answer using retrieved information..."
    
    @staticmethod
    def cross_source_analysis(source_count):
        """Return step and message for cross-source analysis."""
        return "Cross-Source", f"🔄 Analyzing and combining information from {source_count} different sources..."
    
    @staticmethod
    def ai_generation(model_name):
        """Return step and message for AI generation phase."""
        return "AI Generation", f"✍️ Generating response using {model_name}..."
    
    @staticmethod
    def response_success():
        """Return step and message for successful response generation."""
        return "Response", "✅ Response generated successfully", "success"

class UIConfiguration:
    """Configuration for the UI elements including labels, messages, and styling."""
    
    # Page configuration
    PAGE_TITLE = "RAG Chat"
    PAGE_ICON = "💬"
    PAGE_LAYOUT = "wide"
    
    # Main UI elements
    MAIN_TITLE = "💬 RAG Chat with Cohere"
    MAIN_SUBTITLE = "Chat with your data using Cohere's powerful language models"
    WELCOME_MESSAGE = "👋 Welcome to RAG Chat! I can help you find information in your data. Upload documents or ask questions about the compensation database."
    CHAT_INPUT_PLACEHOLDER = "Ask a question about your data..."
    
    # Sidebar elements
    SIDEBAR_UPLOAD_HEADER = "📁 Upload Documents"
    SIDEBAR_UPLOAD_LABEL = "Choose files to upload"
    SIDEBAR_UPLOAD_HELP = "Upload PDF, CSV, TXT files or images to chat with them"
    SIDEBAR_DATA_HEADER = "🗄️ Data Sources"
    SIDEBAR_SETTINGS_HEADER = "⚙️ Settings"
    
    # Buttons and toggles
    PROCESS_FILES_BUTTON = "🔍 Process Files"
    RELOAD_DB_BUTTON = "🔄 Reload Compensation Database"
    CLEAR_CHAT_BUTTON = "🗑️ Clear Chat History"
    WEB_FALLBACK_TOGGLE = "Enable Web Search Fallback"
    WEB_FALLBACK_HELP = "Search the web when no relevant information is found in your data"
    DETAILED_EVALUATION_BUTTON = "🔍 Evaluate"
    
    # Analysis and sources
    SEARCH_ANALYSIS_HEADER = "🔍 Search Analysis"
    SEARCH_ANALYSIS_SUBHEADER = "How your query was processed"
    SOURCES_HEADER = "📚 Sources"
    
    # Evaluation
    EVALUATION_HEADER = "Response Quality Assessment"
    EVALUATION_DIMENSIONS = [
        {"id": "relevance", "name": "Relevance", "description": "How well the response answers the question"},
        {"id": "factual_accuracy", "name": "Accuracy", "description": "Factual correctness based on retrieved information"},
        {"id": "groundedness", "name": "Groundedness", "description": "Stays grounded in the retrieved information"}
    ]
    
    # Icons and visual elements
    SOURCE_ICONS = {
        "database": "🗄️ Compensation Database",
        "uploads": "📄",
        "web_search": "🌐 Web Search"
    }
    STATUS_ICONS = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️"
    }
    
    # Pipeline steps for visual progress indicator
    PIPELINE_STEPS = [
        {"id": "embedding", "name": "Embedding", "icon": "🔮"},
        {"id": "search", "name": "Search", "icon": "🔍"},
        {"id": "reranking", "name": "Reranking", "icon": "🎯"},
        {"id": "generation", "name": "Generation", "icon": "✍️"}
    ]
    
    # Error messages
    NO_API_KEY_ERROR = "Please provide a Cohere API key to use RAG Chat. Set it as COHERE_API_KEY in your environment or .streamlit/secrets.toml."
    NO_RESULTS_ERROR = "I couldn't find relevant information to answer your question. Please try rephrasing or upload additional documents with pertinent information."

def get_user_config_overrides():
    """Get user-defined configuration overrides from environment or secrets."""
    # This would typically read from environment variables or a config file
    return {}

def apply_config_overrides(config_class, overrides):
    """Apply user configuration overrides to the configuration class."""
    if not overrides:
        return
    
    for key, value in overrides.items():
        if hasattr(config_class, key):
            setattr(config_class, key, value)