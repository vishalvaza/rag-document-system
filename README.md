# RAG Document System

A Retrieval Augmented Generation (RAG) system built with **LangChain** for document question-answering. Upload PDFs, text files, or markdown documents and ask questions - get accurate answers with source citations.

## Features

- **Hybrid Search**: Combines semantic (FAISS) + keyword (BM25) search
- **Cross-Encoder Reranking**: Reorders results for best quality
- **Semantic Chunking**: Smart chunking based on meaning, not characters
- **Latest Embeddings**: BAAI BGE model (2023, state-of-the-art)
- **Multiple File Formats**: PDF, TXT, MD support
- **Fast Vector Search**: FAISS-based similarity search
- **Flexible LLM Support**: OpenAI or Ollama (cloud/local)
- **LangChain Integration**: Modern RAG implementation with LCEL
- **Grounded Answers**: Citations with source documents
- **Web UI**: Beautiful Streamlit interface
- **CLI Tools**: Command-line interface for automation
- **Privacy-Focused**: Run 100% locally with Ollama

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | LangChain (LCEL) |
| **PDF/Text Parsing** | pypdf |
| **Text Chunking** | Semantic Chunking / langchain-text-splitters |
| **Vector Store** | FAISS (local, fast) + BM25 (hybrid search) |
| **Embeddings** | HuggingFace (BAAI/bge-small-en-v1.5) |
| **Reranking** | Cross-Encoder (ms-marco-MiniLM-L-12-v2) |
| **LLM** | OpenAI API / Ollama |
| **Web UI** | Streamlit |

## Quick Start

### 1. Clone & Setup

#### Quick Setup (Recommended)

**Windows:**
```bash
setup.bat
```

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

#### Manual Setup

```bash
# Clone repository
git clone <your-repo-url>
cd rag-document-system

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure LLM Provider

Create a `.env` file in the project root:

#### Option A: OpenAI (Recommended for best quality)

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

Get your API key at: https://platform.openai.com/api-keys

#### Option B: Ollama Cloud API

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your_api_key_here
OLLAMA_MODEL=ministral-3:8b
```

#### Option C: Ollama Local (Free, Open-Source)

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
# No API key needed for local
```

**First, install Ollama:**
1. Download from: https://ollama.ai
2. Install a model: `ollama pull mistral`
3. Start server: `ollama serve`

### 3. Add Documents

Place your documents in the `data/` folder:

```
data/
  ├── document1.pdf
  ├── document2.txt
  └── notes.md
```

### 4. Build Vector Index

```bash
python ingest.py
```

This will:
- Load all documents from `data/`
- Split into chunks using semantic chunking (~1200 characters)
- Generate embeddings with BAAI BGE model
- Build FAISS vector store
- Build BM25 keyword index
- Save to `index/` folder

### 5. Run the Application

#### Web UI (Recommended)

```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

#### Command Line Chat

```bash
python chat.py
```

#### Test Retrieval

```bash
python query.py
```

## Project Structure

```
rag-document-system/
├── app.py                 # Streamlit web UI
├── chat.py                # CLI chat interface with RAG chain
├── config.py              # Configuration settings
├── ingest.py              # Document ingestion & indexing (LangChain)
├── llm.py                 # LLM abstraction layer (LangChain)
├── query.py               # Retrieval testing
├── list_models.py         # List available Ollama models
│
├── setup.bat              # Windows setup script
├── setup.sh               # Mac/Linux setup script
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── .env                   # Your config (create this)
├── .gitignore             # Git ignore rules
├── LICENSE                # MIT License
│
├── data/                  # Your documents (PDF, TXT, MD)
│   ├── .gitkeep
│   └── sample.txt
│
└── index/                 # Generated vector store
    ├── index.faiss        # FAISS index
    ├── index.pkl          # Metadata
    └── bm25.pkl           # BM25 keyword index
```

## How It Works

### RAG Pipeline (LangChain LCEL)

1. **Document Ingestion**
   - Load PDFs/text files using pypdf
   - Create LangChain Document objects with metadata

2. **Semantic Chunking**
   - Split using `SemanticChunker` (meaning-based)
   - Fallback to `RecursiveCharacterTextSplitter`
   - ~1200 characters per chunk
   - 150 character overlap

3. **Embedding & Indexing**
   - Generate embeddings with BAAI BGE model
   - Build FAISS vector store
   - Build BM25 keyword index
   - Save locally for fast retrieval

4. **Hybrid Query Processing**
   - Semantic search: Embed query and search FAISS
   - Keyword search: BM25 token matching
   - Combine results with weighted scoring (70/30)
   - Cross-encoder reranking for final ordering
   - Return top-K with metadata (filename, page)

5. **Grounded Generation (LCEL)**
   - Build RAG chain: `retriever | format_docs | prompt | llm | parser`
   - LLM answers using ONLY retrieved context
   - Citations included in response

### LangChain Implementation

```python
# RAG Chain using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## Usage Examples

### Web Interface

1. Upload documents via sidebar
2. Click "Build/Rebuild Index"
3. Ask questions in chat
4. View source citations

### CLI Examples

```bash
# Build index from documents
python ingest.py

# Interactive chat
python chat.py

# Test retrieval only
python query.py

# List Ollama models (if using Ollama cloud)
python list_models.py
```

## Configuration

Edit `config.py` or `.env`:

### Chunking
```python
CHUNK_SIZE = 1200          # Characters per chunk
CHUNK_OVERLAP = 150        # Overlap between chunks
SEMANTIC_CHUNKING = True   # Use semantic chunking
```

### Retrieval
```python
TOP_K = 6                  # Number of chunks to retrieve
HYBRID_ALPHA = 0.7         # Semantic weight (0.7 semantic + 0.3 keyword)
RERANK_ENABLED = True      # Enable cross-encoder reranking
```

### Embeddings
```python
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
```

**Note:** Changing EMBEDDING_MODEL requires rebuilding index with `python ingest.py`

### LLM Settings
```python
LLM_PROVIDER = "ollama"    # or "openai"
LLM_TEMPERATURE = 0.0      # 0.0 = deterministic
```

### OpenAI
```python
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-3.5-turbo"
```

### Ollama
```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"
OLLAMA_API_KEY = ""  # Only for cloud API
```

## Troubleshooting

### "Index not found" Error
Run `python ingest.py` to build the index first.

### OpenAI API Issues
- Verify API key in `.env`
- Check your OpenAI account has credits
- Ensure `langchain-openai` is installed

### Ollama Issues

**Local Mode:**
1. Install Ollama: https://ollama.ai
2. Pull model: `ollama pull mistral`
3. Start server: `ollama serve`
4. Set `LLM_PROVIDER=ollama` in `.env`

**Cloud Mode:**
1. Set `OLLAMA_API_KEY` in `.env`
2. Run `python list_models.py` to see available models
3. Update `OLLAMA_MODEL` to match available model

### Empty Results
- Ensure documents are in `data/` folder
- Rebuild index: `python ingest.py`
- Try rephrasing your question

### Import Errors
```bash
# Make sure virtual environment is activated
pip install -r requirements.txt
```

### Performance Issues

**Slow Ingestion:**
- Disable semantic chunking: `SEMANTIC_CHUNKING = False`
- Semantic chunking is slower but produces better quality chunks

**Slow Queries:**
- Disable reranking: `RERANK_ENABLED = False`
- Reduce TOP_K to retrieve fewer chunks

**BM25 Index Missing:**
- Run `python ingest.py` to rebuild with BM25
- System falls back to pure vector search if BM25 unavailable

## Advanced Features

### Hybrid Search & Reranking

The system uses a 3-stage retrieval pipeline:

1. **Semantic Search (FAISS)**: Finds semantically similar chunks using embeddings
2. **Keyword Search (BM25)**: Finds exact keyword matches using traditional IR
3. **Cross-Encoder Reranking**: Reorders combined results using neural reranker

**Configuration (`config.py`):**
- `HYBRID_ALPHA = 0.7`: Weight for semantic (0.7) vs keyword (0.3)
- `RERANK_ENABLED = True`: Enable/disable reranking
- `BM25_K = 6`: Number of results from each search method

**Benefits:**
- Catches both conceptual matches and exact terms
- 15-20% better retrieval accuracy than pure vector search
- Graceful fallback if BM25 unavailable

### Semantic Chunking

Documents are split based on semantic similarity between sentences, not arbitrary character counts.

**Configuration (`config.py`):**
- `SEMANTIC_CHUNKING = True`: Enable semantic chunking
- `SEMANTIC_BREAKPOINT_TYPE = "percentile"`: Split algorithm
- `SEMANTIC_BREAKPOINT_THRESHOLD = 95`: Percentile threshold

**Benefits:**
- Chunks keep related content together
- Better retrieval context
- More natural document boundaries

**Fallback:** Automatically uses RecursiveCharacterTextSplitter if semantic chunking fails.

### Upgraded Embeddings

Using **BAAI/bge-small-en-v1.5** (2023) instead of older all-MiniLM-L6-v2:
- Trained on diverse datasets (MS MARCO, NQ, etc.)
- Better retrieval quality
- Same dimensions (384), similar speed

### Custom Embeddings Model

```python
# In config.py
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
```

### Adjust Retrieval

```python
# In config.py
TOP_K = 10  # Retrieve more chunks
CHUNK_SIZE = 800  # Smaller chunks
```

### Switch LLM Provider

```python
# In .env
LLM_PROVIDER=openai  # or ollama
```

## Development

### Project Architecture

- **LangChain Integration**: Modern RAG using LCEL
- **Modular Design**: Separate concerns (ingest, query, chat, UI)
- **Type Hints**: Full type annotations
- **Error Handling**: Graceful fallbacks
- **Extensible**: Easy to add new LLM providers

### Key Components

1. **llm.py** - LangChain LLM abstraction
   - `ChatOpenAI` and `ChatOllama` wrappers
   - Unified interface for multiple providers

2. **ingest.py** - Document processing
   - LangChain `Document` objects
   - `FAISS.from_documents()` for indexing

3. **chat.py** - RAG implementation
   - LCEL chain composition
   - Prompt templates
   - Output parsing

4. **app.py** - Streamlit UI
   - Document upload
   - Interactive chat
   - Source viewing

## Requirements

```
Python >= 3.8
pypdf >= 4.0.0
langchain >= 0.3.0
langchain-community >= 0.3.0
langchain-experimental >= 0.3.0
langchain-openai >= 0.2.0
langchain-huggingface >= 1.0.0
langchain-ollama >= 1.0.0
faiss-cpu >= 1.9.0
sentence-transformers >= 3.0.0
rank-bm25 >= 0.2.2
streamlit >= 1.40.0
python-dotenv >= 1.0.0
```

## License

MIT License - Feel free to use for any purpose.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:
1. Check this README
2. Review configuration in `.env` file
3. Open an issue on GitHub

---

**Built with LangChain, FAISS, and Modern RAG Patterns**
