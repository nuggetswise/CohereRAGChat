import streamlit as st
import os
import tempfile
import cohere
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from rank_bm25 import BM25Okapi  # Import BM25 for keyword search
from typing import List, Dict, Any, Optional, Tuple
import json
import pandas as pd
import requests
from urllib.parse import urlparse
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import plotly.express as px
import umap
import altair as alt
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from ragas.metrics import faithfulness, answer_relevancy

# Import custom embedding utilities
from embedding_utils import (
    EnhancedCohereEmbeddings, 
    multi_query_embedding, 
    visualize_embeddings,
    streamlit_embedding_dashboard,
    REDUCED_DIMENSIONS,
    VISUALIZATION_METHODS
)

# Import centralized configuration
from prompts_and_logic import RAGConfiguration, RAGAnalysisSteps, get_user_config_overrides, apply_config_overrides

# Download NLTK data (first time only)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# New OCR imports
try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    st.warning("OCR libraries not installed. Install with: pip install pytesseract pillow pdf2image")

load_dotenv()

st.set_page_config(
    page_title="Cohere Hybrid Chat",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to get API keys
def get_api_keys():
    openai_key = os.environ.get("OPENAI_API_KEY")
    cohere_key = os.environ.get("COHERE_API_KEY")
    
    if not openai_key and hasattr(st, "secrets"):
        if "OPENAI_API_KEY" in st.secrets:
            openai_key = st.secrets["OPENAI_API_KEY"]
        elif "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets.openai:
            openai_key = st.secrets.openai["OPENAI_API_KEY"]
    
    if not cohere_key and hasattr(st, "secrets"):
        if "COHERE_API_KEY" in st.secrets:
            cohere_key = st.secrets["COHERE_API_KEY"]
        elif "cohere" in st.secrets and "COHERE_API_KEY" in st.secrets.cohere:
            cohere_key = st.secrets.cohere["COHERE_API_KEY"]
    
    return openai_key, cohere_key

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'local_vectorstore' not in st.session_state:
        st.session_state.local_vectorstore = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'cohere_datasets' not in st.session_state:
        st.session_state.cohere_datasets = []
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    if 'evaluation_metrics' not in st.session_state:
        st.session_state.evaluation_metrics = {
            'precision': [],
            'recall': [],
            'relevance': []
        }
    if 'search_config' not in st.session_state:
        st.session_state.search_config = {
            'semantic_weight': 0.7,
            'keyword_weight': 0.3,
            'rerank_top_k': 10
        }
    if 'embedding_config' not in st.session_state:
        st.session_state.embedding_config = {
            'model': 'embed-v4.0',
            'dimensions': 1024,
            'use_multi_query': False,
            'query_variations': 3
        }
    if 'document_embeddings' not in st.session_state:
        st.session_state.document_embeddings = None
    if 'document_labels' not in st.session_state:
        st.session_state.document_labels = None

def get_cohere_datasets(cohere_key):
    """Fetch available datasets from Cohere platform."""
    if not cohere_key:
        return []
    
    try:
        co = cohere.Client(api_key=cohere_key)
        response = co.list_datasets()
        return response.datasets if hasattr(response, 'datasets') else []
    except Exception as e:
        st.error(f"Error fetching Cohere datasets: {e}")
        return []

def search_cohere_dataset(dataset_id, query, cohere_key, top_k=5):
    """Search within a Cohere dataset."""
    if not cohere_key or not dataset_id:
        return [], []
    
    try:
        co = cohere.Client(api_key=cohere_key)
        
        # Use Cohere's dataset search functionality
        response = co.chat(
            model="command-r-plus",
            message=f"""Search the dataset for information related to: {query}
            
Please provide relevant information from the dataset that answers the query.""",
            # You can specify the dataset here if the API supports it
            temperature=0.3
        )
        
        return [response.text], ["🔮 Cohere Dataset Search"]
        
    except Exception as e:
        st.warning(f"Cohere dataset search failed: {e}")
        return [], []

def create_dataset_file_for_upload(texts):
    """Create a JSONL file suitable for Cohere dataset upload."""
    data = []
    for i, text in enumerate(texts):
        data.append({
            "id": f"doc_{i}",
            "text": text.page_content,
            "metadata": text.metadata
        })
    
    return data

def process_uploaded_files(uploaded_files, cohere_key):
    """Process uploaded PDF files and create embeddings using Cohere."""
    if not cohere_key:
        st.error("Cohere API key required for document processing")
        return None
    
    all_texts = []
    
    with st.spinner('Processing uploaded files...'):
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    texts = text_splitter.split_documents(documents)
                    
                    for text in texts:
                        text.metadata['source_file'] = uploaded_file.name
                    
                    all_texts.extend(texts)
                    st.session_state.processed_files.append(uploaded_file.name)
                    
                finally:
                    os.unlink(tmp_path)
    
    if all_texts:
        # Get embedding configuration
        embedding_config = st.session_state.embedding_config
        
        # Use enhanced embeddings with custom dimensions
        embeddings = EnhancedCohereEmbeddings(
            model=embedding_config['model'],
            cohere_api_key=cohere_key,
            dimensions=embedding_config['dimensions']
        )
        
        vectorstore = FAISS.from_documents(all_texts, embeddings)
        
        # Store document embeddings and labels for visualization
        doc_embeddings = []
        doc_labels = []
        
        for i, doc in enumerate(all_texts):
            try:
                doc_vector = vectorstore.docstore.search(vectorstore.index_to_docstore_id[i]).embedding
                if doc_vector is not None:
                    doc_embeddings.append(doc_vector)
                    # Create a short label
                    content_preview = doc.page_content[:30].replace("\n", " ")
                    source = doc.metadata.get('source_file', 'unknown')
                    doc_labels.append(f"{source}: {content_preview}...")
            except (KeyError, AttributeError):
                pass
        
        if doc_embeddings:
            st.session_state.document_embeddings = doc_embeddings
            st.session_state.document_labels = doc_labels
        
        # Option to create dataset file for Cohere upload
        dataset_data = create_dataset_file_for_upload(all_texts)
        
        st.success(f"✅ Processed {len(all_texts)} document chunks from {len(uploaded_files)} files")
        
        # Provide download option for Cohere dataset
        if st.button("📥 Download as Cohere Dataset (JSONL)"):
            jsonl_content = "\n".join([json.dumps(item) for item in dataset_data])
            st.download_button(
                label="Download JSONL for Cohere",
                data=jsonl_content,
                file_name="cohere_dataset.jsonl",
                mime="application/json"
            )
            st.info("💡 Upload this file to Cohere Dashboard > Datasets to create an embed dataset!")
        
        return vectorstore
    
    return None

def process_url_content(urls, cohere_key):
    """Process URLs and extract content for embedding."""
    if not cohere_key:
        st.error("Cohere API key required for URL processing")
        return None
    
    all_texts = []
    
    with st.spinner('Extracting content from URLs...'):
        for url in urls:
            try:
                # Validate URL
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    st.warning(f"Invalid URL: {url}")
                    continue
                
                # Load web content
                loader = WebBaseLoader(url)
                documents = loader.load()
                
                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                
                # Add source information
                for text in texts:
                    text.metadata['source_url'] = url
                    text.metadata['source_type'] = 'web'
                
                all_texts.extend(texts)
                st.success(f"✅ Processed: {url}")
                
            except Exception as e:
                st.error(f"Failed to process {url}: {str(e)}")
    
    if all_texts:
        # Get embedding configuration
        embedding_config = st.session_state.embedding_config
        
        # Use enhanced embeddings with custom dimensions
        embeddings = EnhancedCohereEmbeddings(
            model=embedding_config['model'],
            cohere_api_key=cohere_key,
            dimensions=embedding_config['dimensions']
        )
        
        # Create or update FAISS vector store
        if st.session_state.local_vectorstore:
            # Add to existing vectorstore
            st.session_state.local_vectorstore.add_documents(all_texts)
            vectorstore = st.session_state.local_vectorstore
        else:
            # Create new vectorstore
            vectorstore = FAISS.from_documents(all_texts, embeddings)
        
        st.success(f"✅ Processed {len(all_texts)} chunks from {len(urls)} URLs")
        return vectorstore
    
    return None

def build_bm25_index(documents):
    """Build a BM25 index for keyword search."""
    # Extract text content and tokenize
    corpus = [doc.page_content for doc in documents]
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    
    # Create BM25 index
    return BM25Okapi(tokenized_corpus), corpus

def keyword_search(query, bm25_index, corpus, top_k=10):
    """Perform keyword search using BM25."""
    tokenized_query = word_tokenize(query.lower())
    scores = bm25_index.get_scores(tokenized_query)
    
    # Get top-k documents based on BM25 scores
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    return [corpus[i] for i in top_indices], [scores[i] for i in top_indices]

def multi_query_search(query, vectorstore, cohere_key, config=None, top_k=5):
    """Perform search using multiple query variations to improve recall."""
    if not config:
        config = {
            'model': 'embed-v4.0',
            'query_variations': 3
        }
    
    # Create Cohere client
    co = cohere.Client(api_key=cohere_key)
    
    # Generate query variations and embeddings
    query_data = multi_query_embedding(
        query=query,
        client=co,
        model=config['model'],
        variation_count=config['query_variations']
    )
    
    # Collect results from all query variations
    all_results = []
    seen_documents = set()
    
    # Search with each query variation
    for i, variation in enumerate(query_data['variations']):
        # Get the specific embedding for this variation
        query_embedding = query_data['embeddings'][i]
        
        # Search using the embedding directly
        results = vectorstore.similarity_search_by_vector(
            embedding=query_embedding,
            k=top_k
        )
        
        # Add unique results
        for doc in results:
            # Create a simple hash of the document content to check uniqueness
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen_documents:
                seen_documents.add(doc_hash)
                all_results.append(doc)
    
    # If we have too many results, rerank them
    if len(all_results) > top_k and cohere_key:
        try:
            docs_for_rerank = [doc.page_content for doc in all_results]
            
            rerank_response = co.rerank(
                model="rerank-v3.5",
                query=query,
                documents=docs_for_rerank,
                top_n=top_k,
                return_documents=True
            )
            
            # Get reranked results
            final_results = []
            for result in rerank_response.results[:top_k]:
                final_results.append(all_results[result.index])
            
            return final_results, ["🔍 Multi-Query Search + Reranking"]
        except Exception as e:
            st.warning(f"Reranking failed: {e}")
    
    # Return the top results
    return all_results[:top_k], ["🔍 Multi-Query Search"]

def hybrid_search(query, vectorstore, docs, cohere_key, config=None):
    """Perform hybrid search combining semantic and keyword approaches with reranking."""
    if config is None:
        config = {
            'semantic_weight': 0.7,
            'keyword_weight': 0.3,
            'rerank_top_k': 10,
            'use_multi_query': False,
            'query_variations': 3
        }
        
    semantic_weight = config.get('semantic_weight', 0.7)
    keyword_weight = config.get('keyword_weight', 0.3)
    rerank_top_k = config.get('rerank_top_k', 10)
    use_multi_query = config.get('use_multi_query', False)
    
    results = []
    scores = []
    sources = []
    
    # Part 1: Vector Search (Semantic) - with or without multi-query
    try:
        if use_multi_query:
            # Use multi-query approach
            semantic_docs, multi_query_sources = multi_query_search(
                query=query,
                vectorstore=vectorstore,
                cohere_key=cohere_key,
                config=st.session_state.embedding_config,
                top_k=rerank_top_k
            )
            sources.extend(multi_query_sources)
        else:
            # Use regular vector search
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": rerank_top_k}
            )
            
            semantic_docs = retriever.get_relevant_documents(query)
            sources.append("📊 Semantic Search")
            
        if semantic_docs:
            results.extend(semantic_docs)
            # Placeholder scores for semantic results (we'll rerank later)
            scores.extend([1.0] * len(semantic_docs))
    except Exception as e:
        st.warning(f"Semantic search failed: {e}")
    
    # Part 2: Keyword Search (BM25)
    try:
        # Build BM25 index for all documents
        bm25_index, corpus = build_bm25_index(docs)
        
        # Get keyword search results
        keyword_results, keyword_scores = keyword_search(query, bm25_index, docs, top_k=rerank_top_k)
        
        if keyword_results:
            # Filter out duplicates from semantic search
            existing_texts = {doc.page_content for doc in results}
            for doc, score in zip(keyword_results, keyword_scores):
                if doc.page_content not in existing_texts:
                    results.append(doc)
                    scores.append(score * keyword_weight)  # Weight the keyword scores
                    existing_texts.add(doc.page_content)
            
            if "📊 Keyword Search" not in sources:
                sources.append("📊 Keyword Search")
    except Exception as e:
        st.warning(f"Keyword search failed: {e}")
    
    # Part 3: Reranking with Cohere
    if results and cohere_key:
        try:
            co = cohere.Client(api_key=cohere_key)
            docs_for_rerank = [doc.page_content for doc in results]
            
            rerank_response = co.rerank(
                model="rerank-v3.5",  # Upgraded to latest rerank model
                query=query,
                documents=docs_for_rerank,
                top_n=min(rerank_top_k, len(docs_for_rerank)),
                return_documents=True
            )
            
            # Get reranked results with their relevance scores
            reranked_results = []
            for result in rerank_response.results:
                idx = result.index
                reranked_results.append({
                    'document': results[idx],
                    'score': result.relevance_score
                })
            
            # Sort by score and convert back to document list
            reranked_results.sort(key=lambda x: x['score'], reverse=True)
            results = [item['document'] for item in reranked_results]
            
            sources = ["🔍 Hybrid Search + Reranking"]
            
        except Exception as e:
            st.warning(f"Reranking failed: {e}. Using combined results without reranking.")
    
    # If we have no results at all, return empty lists
    if not results:
        return [], []
    
    return results[:rerank_top_k], sources

def visualize_search_results(query, results, method="PCA"):
    """Visualize search results in a 2D plot."""
    if not results:
        return None
    
    # Extract embeddings - this assumes the results have embeddings
    try:
        embeddings = [result._embedding for result in results if hasattr(result, '_embedding')]
        
        if not embeddings:
            return None
        
        # Create labels for the embeddings
        labels = []
        for result in results[:len(embeddings)]:
            # Create a short label
            content_preview = result.page_content[:30].replace("\n", " ")
            source = result.metadata.get('source_file', result.metadata.get('source_url', 'unknown'))
            labels.append(f"{source}: {content_preview}...")
        
        # Add query embedding
        client = cohere.Client(api_key=st.session_state.get('cohere_key'))
        
        # Get query embedding
        embedding_config = st.session_state.embedding_config
        embed_response = client.embed(
            texts=[query],
            model=embedding_config['model'],
            input_type="search_query"
        )
        query_embedding = embed_response.embeddings[0]
        
        # Visualize using the utility function
        viz_result = visualize_embeddings(
            embeddings=embeddings,
            labels=labels,
            method=method,
            title=f"Search Results for '{query}'",
            reference_embedding=query_embedding,
            reference_label=f"Query: {query[:30]}...",
            interactive=True,
        )
        
        return viz_result["scatter_plot"]
    
    except Exception as e:
        st.warning(f"Visualization failed: {e}")
        return None

def evaluate_search_quality(query, relevant_docs, ground_truth=None):
    """Evaluate search quality metrics."""
    try:
        # If we have ground truth (not likely in a live app), we can calculate precision/recall
        if ground_truth:
            # Calculate precision
            relevant_ids = {doc.metadata.get('id', i) for i, doc in enumerate(relevant_docs)}
            ground_truth_ids = {doc.metadata.get('id', i) for i, doc in enumerate(ground_truth)}
            
            precision = len(relevant_ids.intersection(ground_truth_ids)) / len(relevant_ids) if relevant_ids else 0
            recall = len(relevant_ids.intersection(ground_truth_ids)) / len(ground_truth_ids) if ground_truth_ids else 0
            
            return {
                'precision': precision,
                'recall': recall,
                'relevance': None  # We'd need more complex evaluation for relevance
            }
        
        # Without ground truth, estimate relevance score
        # This is a simplified approach - in a real system you'd use more sophisticated metrics
        if relevant_docs:
            # Check if the first doc contains query terms as a simple relevance check
            first_doc_text = relevant_docs[0].page_content.lower()
            query_terms = query.lower().split()
            term_matches = sum(1 for term in query_terms if term in first_doc_text)
            relevance_score = term_matches / len(query_terms) if query_terms else 0
            
            return {
                'precision': None,  # Can't calculate without ground truth
                'recall': None,     # Can't calculate without ground truth
                'relevance': relevance_score
            }
            
        return {
            'precision': 0,
            'recall': 0,
            'relevance': 0
        }
    
    except Exception as e:
        st.warning(f"Evaluation failed: {e}")
        return {
            'precision': 0,
            'recall': 0,
            'relevance': 0
        }

def search_documents(query, vectorstore, cohere_key, all_docs=None):
    """Search documents using hybrid search and reranking."""
    if not vectorstore:
        return [], []
    
    # Get search configuration from session state
    search_config = st.session_state.search_config
    embedding_config = st.session_state.embedding_config
    
    # Combine configurations
    config = {
        **search_config,
        **embedding_config
    }
    
    # If all_docs is not provided, get all documents from the vectorstore
    if all_docs is None:
        # This is a simplification - in practice, you'd need a way to get all docs from vectorstore
        all_docs = vectorstore.similarity_search("", k=1000)  # Get a large number of docs
    
    # Perform hybrid search
    relevant_docs, sources = hybrid_search(
        query, 
        vectorstore, 
        all_docs, 
        cohere_key,
        config
    )
    
    if not relevant_docs:
        return [], []
    
    # Evaluate search quality
    eval_metrics = evaluate_search_quality(query, relevant_docs)
    
    # Store metrics in session state for tracking
    if eval_metrics['relevance'] is not None:
        st.session_state.evaluation_metrics['relevance'].append(eval_metrics['relevance'])
    if eval_metrics['precision'] is not None:
        st.session_state.evaluation_metrics['precision'].append(eval_metrics['precision'])
    if eval_metrics['recall'] is not None:
        st.session_state.evaluation_metrics['recall'].append(eval_metrics['recall'])
    
    return relevant_docs, sources

def web_search_fallback(query):
    """Fallback to web search using DuckDuckGo."""
    try:
        search = DuckDuckGoSearchRun(num_results=3)
        results = search.run(query)
        return results, ["🌐 Web Search (DuckDuckGo)"]
    except Exception as e:
        return f"Web search failed: {e}", ["❌ Search Unavailable"]

def generate_response(query, context_docs, context_text, openai_key, cohere_key):
    """Generate response using OpenAI (primary) or Cohere (fallback)."""
    
    # Prepare context
    if context_docs:
        context = "\n\n".join([doc.page_content for doc in context_docs])
    else:
        context = context_text
    
    # Create a detailed prompt template
    if context:
        prompt_template = f"""
# TASK
Answer the user's question based on the provided context. Be informative, accurate, and helpful.

# CONTEXT
{context}

# QUESTION
{query}

# INSTRUCTIONS
1. Answer ONLY based on the provided context
2. If the context doesn't contain relevant information, say so
3. Avoid making up information not present in the context
4. Provide specific details from the context when possible
5. Format your answer clearly with markdown if helpful

# ANSWER:
"""
    else:
        prompt_template = f"""
# TASK
Answer the user's question based on your knowledge. Be informative, accurate, and helpful.

# QUESTION
{query}

# INSTRUCTIONS
1. If you don't know the answer, say so clearly
2. Format your answer clearly with markdown if helpful

# ANSWER:
"""
    
    # Try OpenAI first
    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_template}],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content, "OpenAI"
        except Exception as e:
            st.warning(f"OpenAI failed: {str(e)[:50]}... Trying Cohere")
    
    # Fallback to Cohere
    if cohere_key:
        try:
            co = cohere.Client(api_key=cohere_key)
            response = co.chat(
                model="command-r-plus",
                message=prompt_template,
                temperature=0.3,
                tools=[
                    {
                        "name": "evaluate_response_quality",
                        "description": "Evaluates the quality and relevance of the response",
                        "parameter_definitions": {
                            "confidence_score": {
                                "description": "A score from 1-10 indicating confidence in the response",
                                "type": "number",
                                "required": True
                            },
                            "reasoning": {
                                "description": "Brief explanation of confidence score",
                                "type": "string",
                                "required": True
                            }
                        }
                    }
                ]
            )
            
            # Extract confidence score if available
            confidence_score = 7.0  # Default
            confidence_reason = ""
            
            if hasattr(response, 'tool_outputs') and response.tool_outputs:
                for tool_output in response.tool_outputs:
                    if tool_output.name == "evaluate_response_quality":
                        confidence_score = tool_output.outputs.get("confidence_score", 7.0)
                        confidence_reason = tool_output.outputs.get("reasoning", "")
            
            return {
                "text": response.text,
                "confidence_score": confidence_score,
                "confidence_reason": confidence_reason
            }, "Cohere"
            
        except Exception as e:
            return f"Error generating response: {e}", "Error"
    
    return "No AI providers available. Please configure API keys.", "Error"

def process_mixed_files(uploaded_files, cohere_key):
    """Process uploaded files (PDFs and images) with OCR support."""
    if not cohere_key:
        st.error("Cohere API key required for document processing")
        return None
    
    all_texts = []
    
    with st.spinner('Processing uploaded files with OCR...'):
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.type == "application/pdf":
                    # Handle PDF files
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Try regular PDF text extraction first
                        loader = PyPDFLoader(tmp_path)
                        documents = loader.load()
                        
                        # If OCR is available and documents are mostly empty, try OCR
                        if OCR_AVAILABLE and documents and all(len(doc.page_content.strip()) < 50 for doc in documents):
                            st.info(f"🔍 Using OCR for scanned PDF: {uploaded_file.name}")
                            # Convert PDF to images and OCR
                            images = pdf2image.convert_from_path(tmp_path)
                            ocr_text = []
                            
                            for i, image in enumerate(images):
                                text = pytesseract.image_to_string(image)
                                if text.strip():
                                    ocr_text.append({
                                        'page_content': text,
                                        'metadata': {
                                            'source_file': uploaded_file.name,
                                            'page': i + 1,
                                            'extraction_method': 'OCR'
                                        }
                                    })
                            
                            # Create Document objects from OCR results
                            from langchain.docstore.document import Document
                            documents = [Document(page_content=item['page_content'], metadata=item['metadata']) for item in ocr_text]
                        
                        # Split text
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, 
                            chunk_overlap=200
                        )
                        texts = text_splitter.split_documents(documents)
                        
                        for text in texts:
                            if 'source_file' not in text.metadata:
                                text.metadata['source_file'] = uploaded_file.name
                        
                        all_texts.extend(texts)
                        st.session_state.processed_files.append(uploaded_file.name)
                        
                    finally:
                        os.unlink(tmp_path)
                
                elif uploaded_file.type in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
                    # Handle image files with OCR
                    if not OCR_AVAILABLE:
                        st.error(f"OCR not available for image: {uploaded_file.name}")
                        continue
                    
                    # Load image and perform OCR
                    image = Image.open(uploaded_file)
                    text = pytesseract.image_to_string(image)
                    
                    if text.strip():
                        from langchain.docstore.document import Document
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source_file': uploaded_file.name,
                                'extraction_method': 'OCR',
                                'file_type': 'image'
                            }
                        )
                        
                        # Split text
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, 
                            chunk_overlap=200
                        )
                        texts = text_splitter.split_documents([doc])
                        all_texts.extend(texts)
                        st.session_state.processed_files.append(uploaded_file.name)
                        st.success(f"🔍 OCR extracted text from: {uploaded_file.name}")
                    else:
                        st.warning(f"No text found in image: {uploaded_file.name}")
                
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.type}")
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    if all_texts:
        # Get embedding configuration
        embedding_config = st.session_state.embedding_config
        
        # Use enhanced embeddings with custom dimensions
        embeddings = EnhancedCohereEmbeddings(
            model=embedding_config['model'],
            cohere_api_key=cohere_key,
            dimensions=embedding_config['dimensions']
        )
        
        vectorstore = FAISS.from_documents(all_texts, embeddings)
        
        # Store document embeddings and labels for visualization
        doc_embeddings = []
        doc_labels = []
        
        for i, doc in enumerate(all_texts):
            try:
                doc_vector = vectorstore.docstore.search(vectorstore.index_to_docstore_id[i]).embedding
                if doc_vector is not None:
                    doc_embeddings.append(doc_vector)
                    # Create a short label
                    content_preview = doc.page_content[:30].replace("\n", " ")
                    source = doc.metadata.get('source_file', 'unknown')
                    doc_labels.append(f"{source}: {content_preview}...")
            except (KeyError, AttributeError):
                pass
        
        if doc_embeddings:
            st.session_state.document_embeddings = doc_embeddings
            st.session_state.document_labels = doc_labels
        
        # Option to create dataset file for Cohere upload
        dataset_data = create_dataset_file_for_upload(all_texts)
        
        st.success(f"✅ Processed {len(all_texts)} document chunks from {len(uploaded_files)} files")
        
        # Provide download option for Cohere dataset
        if st.button("📥 Download as Cohere Dataset (JSONL)"):
            jsonl_content = "\n".join([json.dumps(item) for item in dataset_data])
            st.download_button(
                label="Download JSONL for Cohere",
                data=jsonl_content,
                file_name="cohere_dataset.jsonl",
                mime="application/json"
            )
            st.info("💡 Upload this file to Cohere Dashboard > Datasets to create an embed dataset!")
        
        return vectorstore, all_texts
    
    return None, []

def display_search_metrics():
    """Display search metrics in the UI."""
    metrics = st.session_state.evaluation_metrics
    
    colored_header("📊 Search Quality Metrics", description="", color_name="blue-70")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_relevance = sum(metrics['relevance']) / len(metrics['relevance']) if metrics['relevance'] else 0
        st.metric("Avg. Relevance", f"{avg_relevance:.2f}")
        
    with col2:
        avg_precision = sum(metrics['precision']) / len(metrics['precision']) if metrics['precision'] else 'N/A'
        if avg_precision != 'N/A':
            st.metric("Avg. Precision", f"{avg_precision:.2f}")
        else:
            st.metric("Avg. Precision", "N/A")
            
    with col3:
        avg_recall = sum(metrics['recall']) / len(metrics['recall']) if metrics['recall'] else 'N/A'
        if avg_recall != 'N/A':
            st.metric("Avg. Recall", f"{avg_recall:.2f}")
        else:
            st.metric("Avg. Recall", "N/A")
    
    style_metric_cards()
    
    # Show trend if we have enough data points
    if len(metrics['relevance']) > 2:
        data = pd.DataFrame({
            'Query': range(1, len(metrics['relevance'])+1),
            'Relevance': metrics['relevance']
        })
        
        chart = alt.Chart(data).mark_line().encode(
            x='Query',
            y=alt.Y('Relevance', scale=alt.Scale(domain=[0, 1])),
            tooltip=['Query', 'Relevance']
        ).properties(
            title='Search Relevance Trend',
            width=600,
            height=300
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

def display_embedding_visualization():
    """Display embedding visualization if document embeddings are available."""
    if (st.session_state.document_embeddings is not None and 
        st.session_state.document_labels is not None and
        len(st.session_state.document_embeddings) > 0):
        
        # Use the embedding dashboard from our utility module
        streamlit_embedding_dashboard(
            embeddings=st.session_state.document_embeddings,
            labels=st.session_state.document_labels,
            query=None,  # No active query
            query_embedding=None
        )
    else:
        st.info("No document embeddings available for visualization. Process documents first.")

# Main app
st.title("🔮 Cohere Hybrid Chat")
st.caption("Local files + Cohere datasets + Web search → Intelligent answers")

# Initialize session state
init_session_state()

# Get API keys
openai_key, cohere_key = get_api_keys()
if cohere_key:
    st.session_state.cohere_key = cohere_key

# API Key configuration
if not openai_key and not cohere_key:
    with st.expander("⚙️ Configure API Keys"):
        col1, col2 = st.columns(2)
        with col1:
            openai_key = st.text_input("OpenAI API Key:", type="password")
        with col2:
            cohere_key = st.text_input("Cohere API Key:", type="password")
        
        if openai_key or cohere_key:
            st.session_state.cohere_key = cohere_key
            st.success("API key(s) configured for this session")

# Sidebar configuration
with st.sidebar:
    st.header("📊 Data Sources")
    
    # Embedding configuration expander
    with st.expander("🧠 Embedding Settings", expanded=False):
        st.subheader("Embedding Configuration")
        
        # Model selection
        embed_model = st.selectbox(
            "Embedding Model",
            options=["embed-english-v3.0", "embed-v4.0", "embed-multilingual-v3.0", "embed-multilingual-v4.0"],
            index=1,  # Default to embed-v4.0
            help="The Cohere embedding model to use"
        )
        
        # Embedding dimensions
        embed_dimensions = st.select_slider(
            "Embedding Dimensions",
            options=REDUCED_DIMENSIONS,
            value=st.session_state.embedding_config['dimensions'],
            help="Number of dimensions for the embeddings (lower values are faster but less accurate)"
        )
        
        # Multi-query option
        use_multi_query = st.checkbox(
            "Use Multi-Query Expansion", 
            value=st.session_state.embedding_config['use_multi_query'],
            help="Generate multiple variations of the query to improve retrieval"
        )
        
        # Query variations
        query_variations = 3
        if use_multi_query:
            query_variations = st.slider(
                "Query Variations",
                min_value=2,
                max_value=5,
                value=3,
                help="Number of query variations to generate"
            )
        
        # Update embedding configuration
        st.session_state.embedding_config = {
            'model': embed_model,
            'dimensions': embed_dimensions,
            'use_multi_query': use_multi_query,
            'query_variations': query_variations
        }
    
    # Search configuration expander
    with st.expander("🔧 Search Settings", expanded=False):
        st.subheader("Hybrid Search Configuration")
        
        # Semantic vs Keyword weights
        semantic_weight = st.slider(
            "Semantic Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.search_config['semantic_weight'],
            help="Balance between semantic (embedding) and keyword (BM25) search"
        )
        
        # Calculate keyword weight as complement
        keyword_weight = 1.0 - semantic_weight
        st.info(f"Keyword Weight: {keyword_weight:.2f}")
        
        # Reranking parameters
        rerank_top_k = st.slider(
            "Rerank Top K", 
            min_value=5, 
            max_value=30, 
            value=st.session_state.search_config['rerank_top_k'],
            help="Number of top results to rerank"
        )
        
        # Update search configuration
        st.session_state.search_config = {
            'semantic_weight': semantic_weight,
            'keyword_weight': keyword_weight,
            'rerank_top_k': rerank_top_k
        }
    
    # Tab selection for different data sources
    tab1, tab2, tab3 = st.tabs(["📁 Local Files", "🔮 Cohere Datasets", "📊 Visualize"])
    
    with tab1:
        st.subheader("Upload Files")
        
        # Check OCR availability
        if OCR_AVAILABLE:
            st.success("🔍 OCR enabled - supports images and handwritten text!")
            file_types = ["pdf", "jpg", "jpeg", "png", "webp"]
            help_text = "Upload PDFs, images, or scanned documents. OCR will extract text from images and handwritten notes."
        else:
            st.warning("📄 Text-only mode - install OCR for image support")
            file_types = ["pdf"]
            help_text = "Upload PDF files for text extraction. For image support, install: pip install pytesseract pillow pdf2image"
        
        uploaded_files = st.file_uploader(
            "Upload Files",
            type=file_types,
            accept_multiple_files=True,
            help=help_text
        )
        
        # Add URL input section
        st.subheader("Add URLs")
        url_input = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            help="Add web pages to process alongside files"
        )
        
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if uploaded_files and cohere_key:
                if st.button("Process Files (with OCR)", type="primary"):
                    vectorstore, all_docs = process_mixed_files(uploaded_files, cohere_key)
                    if vectorstore:
                        st.session_state.local_vectorstore = vectorstore
                        st.session_state.all_docs = all_docs
                        st.rerun()
        
        with col2:
            if urls and cohere_key:
                if st.button("Process URLs", type="secondary"):
                    vectorstore = process_url_content(urls, cohere_key)
                    if vectorstore:
                        st.session_state.local_vectorstore = vectorstore
                        st.rerun()
        
        # Process both together
        if (uploaded_files or urls) and cohere_key:
            if st.button("Process All Sources", type="primary", use_container_width=True):
                combined_vectorstore = None
                all_docs = []
                
                # Process files first
                if uploaded_files:
                    file_vectorstore, file_docs = process_mixed_files(uploaded_files, cohere_key)
                    if file_vectorstore:
                        combined_vectorstore = file_vectorstore
                        all_docs.extend(file_docs)
                
                # Process URLs and combine
                if urls:
                    url_vectorstore = process_url_content(urls, cohere_key)
                    if url_vectorstore:
                        if combined_vectorstore:
                            # Combine with existing vectorstore
                            combined_vectorstore.merge_from(url_vectorstore)
                        else:
                            combined_vectorstore = url_vectorstore
                
                if combined_vectorstore:
                    st.session_state.local_vectorstore = combined_vectorstore
                    st.session_state.all_docs = all_docs
                    st.rerun()
        
        # Display processed files
        if st.session_state.processed_files:
            st.subheader("📄 Processed Files")
            for file in st.session_state.processed_files:
                st.write(f"✅ {file}")
            
            if st.button("🗑️ Clear All Files"):
                st.session_state.local_vectorstore = None
                st.session_state.processed_files = []
                st.session_state.all_docs = []
                st.session_state.document_embeddings = None
                st.session_state.document_labels = None
                st.rerun()
    
    with tab2:
        st.subheader("🔮 Cohere Datasets")
        
        if cohere_key:
            if st.button("🔄 Refresh Datasets"):
                st.session_state.cohere_datasets = get_cohere_datasets(cohere_key)
                st.rerun()
            
            if not st.session_state.cohere_datasets:
                st.session_state.cohere_datasets = get_cohere_datasets(cohere_key)
            
            if st.session_state.cohere_datasets:
                dataset_names = [f"{ds.name} (ID: {ds.id})" for ds in st.session_state.cohere_datasets]
                selected_idx = st.selectbox(
                    "Select Dataset:",
                    range(len(dataset_names)),
                    format_func=lambda x: dataset_names[x],
                    index=0 if st.session_state.selected_dataset is None else st.session_state.selected_dataset
                )
                st.session_state.selected_dataset = selected_idx
                
                selected_dataset = st.session_state.cohere_datasets[selected_idx]
                st.info(f"📊 Selected: {selected_dataset.name}")
                st.caption(f"Type: {getattr(selected_dataset, 'type', 'N/A')} | Created: {getattr(selected_dataset, 'created_at', 'N/A')}")
            else:
                st.info("No datasets found. Create datasets in Cohere Dashboard.")
                st.markdown("💡 **Tip**: Upload the JSONL file from 'Local Files' tab to create a dataset!")
        else:
            st.warning("Cohere API key required to access datasets")
    
    with tab3:
        st.subheader("📊 Embedding Visualization")
        
        if st.button("Show Embedding Dashboard", use_container_width=True):
            st.session_state.show_embedding_viz = True
        
        # Visualization method selection
        viz_method = st.selectbox(
            "Visualization Method:", 
            VISUALIZATION_METHODS,
            index=0
        )
        st.session_state.viz_method = viz_method
    
    # Search strategy selection
    st.header("🔍 Search Strategy")
    search_options = []
    
    if st.session_state.local_vectorstore:
        search_options.append("📄 Local Documents")
    
    if cohere_key and st.session_state.cohere_datasets and st.session_state.selected_dataset is not None:
        search_options.append("🔮 Cohere Dataset")
    
    search_options.extend(["🌐 Web Search", "🔄 All Sources"])
    
    search_strategy = st.selectbox(
        "Choose search approach:",
        search_options,
        index=len(search_options)-1 if len(search_options) > 1 else 0
    )

# Main interface with tabs
tab1, tab2 = st.tabs(["💬 Chat", "📊 Embedding Analysis"])

# Chat interface tab
with tab1:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"• {source}")
    
    # Show search metrics
    if st.session_state.evaluation_metrics['relevance']:
        display_search_metrics()
    
    # Chat input
    if query := st.chat_input("Ask a question..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                context_docs = []
                context_text = ""
                sources = []
                
                # Execute search strategy
                if search_strategy == "📄 Local Documents" and st.session_state.local_vectorstore:
                    context_docs, sources = search_documents(
                        query, 
                        st.session_state.local_vectorstore, 
                        cohere_key,
                        st.session_state.get('all_docs', [])
                    )
                    
                    # Visualize search results
                    if context_docs:
                        with st.expander("🔍 View Search Visualization", expanded=False):
                            fig = visualize_search_results(
                                query, 
                                context_docs, 
                                method=st.session_state.get('viz_method', 'UMAP')
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Visualization not available for these results")
                
                elif search_strategy == "🔮 Cohere Dataset" and cohere_key and st.session_state.selected_dataset is not None:
                    selected_dataset = st.session_state.cohere_datasets[st.session_state.selected_dataset]
                    context_text, dataset_sources = search_cohere_dataset(selected_dataset.id, query, cohere_key)
                    sources.extend(dataset_sources)
                
                elif search_strategy == "🌐 Web Search":
                    context_text, web_sources = web_search_fallback(query)
                    sources.extend(web_sources)
                
                elif search_strategy == "🔄 All Sources":
                    # Search local documents
                    if st.session_state.local_vectorstore:
                        local_docs, local_sources = search_documents(
                            query, 
                            st.session_state.local_vectorstore, 
                            cohere_key,
                            st.session_state.get('all_docs', [])
                        )
                        context_docs.extend(local_docs)
                        sources.extend(local_sources)
                    
                    # Search Cohere dataset
                    if cohere_key and st.session_state.selected_dataset is not None:
                        selected_dataset = st.session_state.cohere_datasets[st.session_state.selected_dataset]
                        dataset_context, dataset_sources = search_cohere_dataset(selected_dataset.id, query, cohere_key)
                        if isinstance(dataset_context, list):
                            context_text += "\n".join(dataset_context)
                        else:
                            context_text += dataset_context
                        sources.extend(dataset_sources)
                    
                    # Web search fallback if no local results
                    if not context_docs and not context_text:
                        fallback_context, web_sources = web_search_fallback(query)
                        context_text += fallback_context
                        sources.extend(web_sources)
                
                # Generate response
                response, model_used = generate_response(query, context_docs, context_text, openai_key, cohere_key)
                
                # Handle different response formats
                if isinstance(response, dict):
                    response_text = response.get("text", "")
                    confidence_score = response.get("confidence_score", 7.0)
                    confidence_reason = response.get("confidence_reason", "")
                    
                    st.write(response_text)
                    
                    with st.expander("Response Quality Assessment", expanded=False):
                        st.write(f"**Confidence Score:** {confidence_score}/10")
                        st.write(f"**Reasoning:** {confidence_reason}")
                else:
                    response_text = response
                    st.write(response_text)
                
                # Show sources
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write(f"• {source}")
                
                st.caption(f"🤖 Generated by: {model_used}")
        
        # Add assistant response to chat history
        if isinstance(response, dict):
            response_text = response.get("text", "")
        else:
            response_text = response
            
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response_text,
            "sources": sources
        })

# Embedding Analysis tab
with tab2:
    st.header("📊 Document Embedding Analysis")
    
    if (st.session_state.document_embeddings is not None and 
        st.session_state.document_labels is not None and
        len(st.session_state.document_embeddings) > 0):
        
        display_embedding_visualization()
    else:
        st.info("Upload and process documents to visualize their embeddings.")
        st.markdown("""
        ### How to use embedding visualization:
        1. Go to the **Local Files** tab in the sidebar
        2. Upload documents (PDF, text, or images)
        3. Process the files
        4. Return to this tab to see visualizations
        
        ### Benefits of embedding visualization:
        - Understand how your documents relate to each other
        - Identify clusters of similar content
        - Spot outliers that might need attention
        - Visualize how queries relate to your documents
        """)

# Footer
st.markdown("---")
st.markdown("🔮 **Cohere Hybrid Chat** - Combining local knowledge with cloud intelligence")