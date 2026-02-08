"""Chat module with grounded generation using LangChain RAG."""
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from query import HybridRetriever

from llm import get_llm, list_available_providers, get_llm_name
from config import TOP_K


# RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the provided context documents.

IMPORTANT RULES:
1. Answer using ONLY information from the context below
2. If the answer is not in the context, say "I cannot find this information in the provided documents."
3. Provide citations by referring to the source documents
4. Be specific and quote relevant parts when possible
5. If multiple sources support your answer, mention them all

Context:
{context}

Question: {question}

Answer (with citations):"""


class GroundedChatbot:
    """Chatbot that answers questions using retrieved document context with LangChain."""

    def __init__(self, provider_name: str = None):
        """
        Initialize chatbot with specified LLM provider.

        Args:
            provider_name: "openai" or "ollama". If None, uses config setting.
        """
        print("Initializing chatbot...")

        # Initialize hybrid retriever
        print("Initializing hybrid retriever...")
        hybrid_retriever_instance = HybridRetriever()

        # Store reference for direct use
        self._hybrid_retriever = hybrid_retriever_instance

        # Wrap as LangChain retriever for LCEL chain
        self.retriever = hybrid_retriever_instance.vectorstore.as_retriever(
            search_kwargs={"k": TOP_K}
        )

        # Initialize LLM
        try:
            self.llm = get_llm(provider_name)
            print(f"[OK] Using LLM: {get_llm_name(provider_name)}")
            self.llm_available = True
        except Exception as e:
            print(f"Warning: {e}")
            print("Running in retrieval-only mode (no LLM generation)")
            self.llm = None
            self.llm_available = False

        # Create RAG chain using LangChain Expression Language (LCEL)
        if self.llm_available:
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

            def format_docs(docs):
                """Format documents into context string."""
                return "\n\n---\n\n".join([doc.page_content for doc in docs])

            # Build RAG chain
            self.rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            self.rag_chain = None

    def generate_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate an answer to the query using RAG.

        Args:
            query: User question

        Returns:
            Dictionary with answer and source documents
        """
        # Retrieve source documents using hybrid retriever
        docs = self._hybrid_retriever.retrieve(query, k=TOP_K)

        if not self.llm_available:
            # Fallback: just retrieve documents
            answer = "LLM not configured. Here are the relevant document excerpts:\n\n"
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                answer += f"\n{i}. [{metadata.get('filename', 'Unknown')} p.{metadata.get('page', '?')}]\n"
                answer += f"{doc.page_content[:200]}...\n"

            return {
                "answer": answer,
                "source_documents": docs,
                "query": query
            }

        # Use RAG chain
        print("Generating answer...")
        answer = self.rag_chain.invoke(query)

        return {
            "answer": answer,
            "source_documents": docs,
            "query": query
        }

    def format_response(self, response: Dict[str, Any]) -> str:
        """Format the response for display."""
        output = []
        output.append("="*60)
        output.append("ANSWER:")
        output.append("="*60)
        output.append(response["answer"])
        output.append("\n" + "="*60)
        output.append("SOURCES:")
        output.append("="*60)

        for i, doc in enumerate(response["source_documents"], 1):
            metadata = doc.metadata
            output.append(f"\n{i}. {metadata.get('filename', 'Unknown')} (page {metadata.get('page', '?')})")
            output.append(f"  Preview: {doc.page_content[:150]}...")

        return "\n".join(output)


def main():
    """Interactive chat interface."""
    print("="*60)
    print("Document QA Chatbot (LangChain RAG)")
    print("="*60)

    print("\nChecking available LLM providers...")
    available = list_available_providers()
    for provider, is_available in available.items():
        status = "[OK] Available" if is_available else "[FAIL] Not available"
        print(f"  {provider}: {status}")

    try:
        chatbot = GroundedChatbot()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease run 'python ingest.py' first to build the index.")
        return

    print("\n" + "="*60)
    print("Ask questions about your documents. Type 'quit' to exit.")
    print("="*60)

    while True:
        query = input("\nYour question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            response = chatbot.generate_answer(query)
            print("\n" + chatbot.format_response(response))
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
