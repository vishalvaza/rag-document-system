#!/bin/bash

echo "================================================"
echo "RAG Document QA - Setup Script"
echo "================================================"
echo ""

echo "Creating virtual environment..."
python3 -m venv .venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure your LLM:"
echo "   cp .env.example .env"
echo "   nano .env  # or use your favorite editor"
echo "   - For OpenAI: Add OPENAI_API_KEY"
echo "   - For Ollama: Set OLLAMA_BASE_URL and OLLAMA_MODEL"
echo ""
echo "2. Add your documents to the data/ folder"
echo ""
echo "3. Run: python ingest.py"
echo ""
echo "4. Run: streamlit run app.py"
echo ""
echo "To activate the environment in the future, run:"
echo "   source .venv/bin/activate"
echo ""
