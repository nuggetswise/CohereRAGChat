#!/bin/bash
echo "🧼 Resetting virtual environment..."

# Remove existing virtual environment
rm -rf .venv

# Create new virtual environment
python3 -m venv .venv
source .venv/bin/activate

echo "📦 Installing pinned dependencies..."
pip install --upgrade pip

# Install dependencies in order to avoid conflicts
echo "Installing core libraries..."
pip install pydantic==1.10.13
pip install langsmith>=0.1.17,<0.2.0
pip install langchain-core==0.1.42
pip install langchain==0.1.13

echo "Installing AI providers..."
pip install cohere==5.15.0
pip install openai==1.38.0

echo "Installing LangChain integrations..."
pip install langchain-community==0.0.29
pip install langchain-cohere==0.1.5
pip install langchain-openai==0.1.6

echo "Installing vector database..."
pip install qdrant-client==1.7.3
pip install langchain-qdrant==0.1.0

echo "Installing Streamlit and utilities..."
pip install streamlit==1.33.0
pip install tiktoken==0.7.0
pip install python-dotenv==1.0.1
pip install pandas==2.0.3
pip install duckduckgo-search==5.3.0

echo "Installing OCR dependencies..."
pip install pytesseract==0.3.10 \
  Pillow==10.0.1 \
  pdf2image==1.17.0 \
  python-docx==1.1.0

echo "Installing GPT4All for local embeddings..."
pip install gpt4all==2.8.2

echo "✅ Installation complete!"
echo "🚀 Activate with: source .venv/bin/activate"
echo "🚀 Run with: streamlit run pages/RAG_Chat.py"
echo ""
echo "🔧 Installed packages for stability:"
echo "   - pydantic v1.10.13 (avoiding v2 conflicts)"
echo "   - langchain v0.1.13 (stable release)"
echo "   - Compatible langsmith version"
echo "   - All integrations pinned to working versions"