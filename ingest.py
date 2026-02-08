"""Document ingestion and indexing module using LangChain."""
from pathlib import Path
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pickle
import re

from config import (
    DATA_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, SUPPORTED_EXTENSIONS,
    SEMANTIC_CHUNKING, SEMANTIC_BREAKPOINT_TYPE, SEMANTIC_BREAKPOINT_THRESHOLD,
    BM25_INDEX_PATH
)


class DocumentIngestor:
    """Handles document loading, chunking, and indexing using LangChain."""

    def __init__(self):
        print("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Try semantic chunking first, fallback to recursive
        if SEMANTIC_CHUNKING:
            try:
                print("Using semantic chunking...")
                self.text_splitter = SemanticChunker(
                    self.embeddings,
                    breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
                    breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD
                )
                self.chunker_type = "semantic"
            except Exception as e:
                print(f"Warning: Semantic chunker failed ({e}), using RecursiveCharacterTextSplitter")
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                self.chunker_type = "recursive"
        else:
            print("Using recursive character text splitting...")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            self.chunker_type = "recursive"

        self.documents: List[Document] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25 (lowercase + word boundaries)."""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def load_pdf(self, file_path: Path) -> List[Document]:
        """Load PDF and extract text per page as LangChain Documents."""
        docs = []
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                            "page": page_num,
                            "doc_id": file_path.stem
                        }
                    )
                    docs.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        return docs

    def load_text_file(self, file_path: Path) -> List[Document]:
        """Load plain text or markdown file as LangChain Document."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "page": 1,
                    "doc_id": file_path.stem
                }
            )
            return [doc]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def load_documents(self) -> None:
        """Load all supported documents from data directory."""
        print(f"Loading documents from {DATA_DIR}...")
        self.documents = []

        for file_path in DATA_DIR.rglob("*"):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                print(f"Loading: {file_path.name}")
                if file_path.suffix.lower() == ".pdf":
                    self.documents.extend(self.load_pdf(file_path))
                else:
                    self.documents.extend(self.load_text_file(file_path))

        print(f"Loaded {len(self.documents)} document pages/sections")

    def build_bm25_index(self, chunks: List[Document]) -> None:
        """Build and save BM25 index from document chunks."""
        print("Building BM25 index...")

        # Tokenize all chunks
        tokenized_corpus = []
        for chunk in tqdm(chunks, desc="Tokenizing for BM25"):
            tokens = self._tokenize(chunk.page_content)
            tokenized_corpus.append(tokens)

        # Create BM25 index
        bm25 = BM25Okapi(tokenized_corpus)

        # Save BM25 data with chunks for mapping
        bm25_data = {
            'bm25': bm25,
            'tokenized_corpus': tokenized_corpus,
            'chunks': chunks  # Store chunks for retrieval
        }

        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump(bm25_data, f)

        print(f"[OK] BM25 index saved to: {BM25_INDEX_PATH}")

    def build_index(self) -> FAISS:
        """Build and save FAISS vector store using LangChain."""
        if not self.documents:
            print("No documents to index. Run load_documents() first.")
            return None

        # Split documents into chunks
        print("Chunking documents...")
        chunks = []
        for doc in tqdm(self.documents, desc="Splitting"):
            doc_chunks = self.text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)

        print(f"Created {len(chunks)} chunks")

        # Create FAISS vector store
        print("Building FAISS vector store with embeddings...")
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        # Save vector store
        print(f"Saving vector store to {INDEX_DIR}...")
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))

        # Build BM25 index with same chunks
        self.build_bm25_index(chunks)

        print(f"[OK] Index built successfully!")
        print(f"  - {len(chunks)} chunks indexed")
        print(f"  - FAISS saved to: {INDEX_DIR}")
        print(f"  - BM25 saved to: {BM25_INDEX_PATH}")

        return vectorstore


def main():
    """Main ingestion pipeline."""
    print("="*60)
    print("Document Ingestion Pipeline (LangChain)")
    print("="*60)

    ingestor = DocumentIngestor()

    # Load documents
    ingestor.load_documents()

    if not ingestor.documents:
        print("\n[WARNING] No documents found in data/ directory!")
        print("Please add PDF, TXT, or MD files to the data/ directory.")
        return

    # Build vector store
    vectorstore = ingestor.build_index()

    if vectorstore:
        print("\n[OK] Ingestion complete! You can now:")
        print("  - Run: python chat.py")
        print("  - Run: streamlit run app.py")


if __name__ == "__main__":
    main()
