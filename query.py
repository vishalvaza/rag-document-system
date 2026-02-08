"""Query and retrieval module using Hybrid Search (FAISS + BM25) with reranking."""
from typing import List, Tuple, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import pickle
import re
import numpy as np

from config import (
    EMBEDDING_MODEL, INDEX_DIR, TOP_K, BM25_INDEX_PATH,
    BM25_K, HYBRID_ALPHA, RERANK_MODEL, RERANK_TOP_K, RERANK_ENABLED
)


class HybridRetriever:
    """Hybrid search (vector + BM25) with cross-encoder reranking."""

    def __init__(self):
        print("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.bm25 = None
        self.chunks = None
        self.reranker = None

        self.load_index()
        self.load_bm25_index()

        if RERANK_ENABLED:
            self.load_reranker()

    def load_index(self) -> None:
        """Load FAISS vector store."""
        if not INDEX_DIR.exists():
            raise FileNotFoundError(
                f"Index not found at {INDEX_DIR}. "
                "Please run ingest.py first to build the index."
            )

        print(f"Loading vector store from {INDEX_DIR}...")
        self.vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"[OK] Vector store loaded successfully")

    def load_bm25_index(self) -> None:
        """Load BM25 index if available."""
        if not BM25_INDEX_PATH.exists():
            print(f"[WARNING] BM25 index not found at {BM25_INDEX_PATH}")
            print("Falling back to pure vector search. Run ingest.py to rebuild with BM25.")
            return

        print(f"Loading BM25 index from {BM25_INDEX_PATH}...")
        with open(BM25_INDEX_PATH, 'rb') as f:
            bm25_data = pickle.load(f)

        self.bm25 = bm25_data['bm25']
        self.chunks = bm25_data['chunks']

        print(f"[OK] BM25 index loaded successfully ({len(self.chunks)} chunks)")

    def load_reranker(self) -> None:
        """Load cross-encoder reranking model."""
        try:
            print(f"Loading reranker: {RERANK_MODEL}...")
            self.reranker = CrossEncoder(RERANK_MODEL)
            print("[OK] Reranker loaded successfully")
        except Exception as e:
            print(f"[WARNING] Failed to load reranker: {e}")
            print("Continuing without reranking.")
            self.reranker = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 (must match ingestion)."""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] using min-max."""
        if not scores or len(scores) == 1:
            return [1.0] * len(scores)

        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()

        if max_score - min_score == 0:
            return [1.0] * len(scores)

        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()

    def retrieve(self, query: str, k: int = TOP_K) -> List[Document]:
        """
        Retrieve top-k chunks using hybrid search + reranking.

        Falls back to pure vector search if BM25 unavailable.
        """
        # Fallback if no BM25
        if self.bm25 is None:
            return self.vectorstore.similarity_search(query, k=k)

        # Step 1: Vector search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=BM25_K)

        # Step 2: BM25 search
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:BM25_K]

        # Step 3: Combine and deduplicate
        combined_docs: Dict[str, Tuple[Document, float]] = {}

        # Add vector results (FAISS returns distances, lower = better)
        vector_scores_list = [score for _, score in vector_results]
        normalized_vector_scores = self._normalize_scores(vector_scores_list)

        for i, (doc, _) in enumerate(vector_results):
            # Invert since lower FAISS distance = better
            normalized_score = 1.0 - normalized_vector_scores[i]
            doc_key = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}_{hash(doc.page_content)}"

            if doc_key not in combined_docs:
                combined_docs[doc_key] = (doc, HYBRID_ALPHA * normalized_score)

        # Add BM25 results
        bm25_scores_subset = [bm25_scores[i] for i in top_bm25_indices]
        normalized_bm25_scores = self._normalize_scores(bm25_scores_subset)

        for i, idx in enumerate(top_bm25_indices):
            if idx < len(self.chunks):
                doc = self.chunks[idx]
                doc_key = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}_{hash(doc.page_content)}"

                if doc_key in combined_docs:
                    # Add BM25 score to existing
                    existing_doc, existing_score = combined_docs[doc_key]
                    combined_docs[doc_key] = (existing_doc, existing_score + (1 - HYBRID_ALPHA) * normalized_bm25_scores[i])
                else:
                    # New document from BM25
                    combined_docs[doc_key] = (doc, (1 - HYBRID_ALPHA) * normalized_bm25_scores[i])

        # Sort by combined score
        sorted_docs = sorted(combined_docs.values(), key=lambda x: x[1], reverse=True)

        # Get top 2*k candidates for reranking
        candidates = [doc for doc, _ in sorted_docs[:2 * k]]

        # Step 4: Rerank if enabled
        if self.reranker is not None and len(candidates) > 0:
            return self._rerank(query, candidates, k)

        return candidates[:k]

    def _rerank(self, query: str, documents: List[Document], k: int) -> List[Document]:
        """Rerank documents using cross-encoder."""
        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Sort by reranking score
        doc_score_pairs = list(zip(documents, rerank_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in doc_score_pairs[:k]]

    def retrieve_with_scores(self, query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
        """Retrieve with scores (compatibility method)."""
        docs = self.retrieve(query, k)
        return [(doc, 0.0) for doc in docs]

    def format_results(self, results: List[Document]) -> str:
        """Format retrieval results for display."""
        output = []
        for i, doc in enumerate(results, 1):
            metadata = doc.metadata
            output.append(f"\n{'='*60}")
            output.append(f"Result {i} - [{metadata.get('filename', 'Unknown')} p.{metadata.get('page', '?')}]")
            output.append(f"{'-'*60}")
            content = doc.page_content
            output.append(content[:300] + "..." if len(content) > 300 else content)
        return "\n".join(output)


# Backward compatibility
class VectorRetriever(HybridRetriever):
    """Alias for backward compatibility."""
    pass


def main():
    """Test hybrid retrieval."""
    retriever = HybridRetriever()

    print("\n" + "="*60)
    print("Document QA - Hybrid Retrieval Test")
    print("="*60)

    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        print(f"\nSearching for: '{query}'")
        results = retriever.retrieve(query)
        print(retriever.format_results(results))


if __name__ == "__main__":
    main()
