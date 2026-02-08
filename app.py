"""Streamlit web interface for RAG Document QA system (LangChain)."""
import streamlit as st
from pathlib import Path
import sys

from config import DATA_DIR, INDEX_DIR, LLM_PROVIDER
from ingest import DocumentIngestor
from chat import GroundedChatbot
from llm import list_available_providers


# Page configuration
st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide"
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index_built" not in st.session_state:
        st.session_state.index_built = INDEX_DIR.exists() and (INDEX_DIR / "index.faiss").exists()
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = LLM_PROVIDER
    if "available_providers" not in st.session_state:
        st.session_state.available_providers = list_available_providers()


def sidebar():
    """Render sidebar with document management."""
    with st.sidebar:
        st.title("📚 Document Management")

        # Status
        if st.session_state.index_built:
            st.success("✓ Index is ready")
        else:
            st.warning("⚠ Index not built yet")

        st.divider()

        # Upload documents
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Upload documents to add them to the knowledge base"
        )

        if uploaded_files:
            if st.button("Save Uploaded Files", type="primary"):
                with st.spinner("Saving files..."):
                    for uploaded_file in uploaded_files:
                        file_path = DATA_DIR / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    st.success(f"Saved {len(uploaded_files)} file(s)")

        st.divider()

        # Build index
        st.subheader("Build Index")

        # Count documents
        doc_count = sum(1 for f in DATA_DIR.rglob("*") if f.suffix.lower() in {".pdf", ".txt", ".md"})
        st.info(f"Documents in data/: {doc_count}")

        if st.button("🔨 Build/Rebuild Index", type="primary", disabled=doc_count == 0):
            with st.spinner("Building index... This may take a few minutes."):
                try:
                    ingestor = DocumentIngestor()
                    ingestor.load_documents()
                    vectorstore = ingestor.build_index()

                    if vectorstore:
                        st.session_state.index_built = True
                        st.session_state.chatbot = None  # Force reload
                        st.success("✓ Index built successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error building index: {e}")

        if doc_count == 0:
            st.warning("Add documents to data/ folder or upload files above")

        st.divider()

        # LLM Provider Selection
        st.subheader("🤖 LLM Provider")

        # Show available providers
        available = st.session_state.available_providers
        provider_options = []
        provider_labels = {}

        if available.get("openai"):
            provider_options.append("openai")
            provider_labels["openai"] = "OpenAI (GPT models)"
        else:
            provider_labels["openai"] = "OpenAI (not configured)"

        if available.get("ollama"):
            provider_options.append("ollama")
            provider_labels["ollama"] = "Ollama (Local models)"
        else:
            provider_labels["ollama"] = "Ollama (not available)"

        if provider_options:
            # Default to first available or current selection
            default_idx = 0
            if st.session_state.llm_provider in provider_options:
                default_idx = provider_options.index(st.session_state.llm_provider)

            selected_provider = st.selectbox(
                "Select LLM Provider",
                options=provider_options,
                format_func=lambda x: provider_labels[x],
                index=default_idx,
                help="Choose between OpenAI API or local Ollama models"
            )

            # Update if changed
            if selected_provider != st.session_state.llm_provider:
                st.session_state.llm_provider = selected_provider
                st.session_state.chatbot = None  # Force reload
                st.success(f"Switched to {provider_labels[selected_provider]}")
        else:
            st.error("No LLM providers available. Configure OpenAI API key or install Ollama.")
            st.info(
                "**Setup Options:**\n"
                "- **OpenAI**: Add OPENAI_API_KEY to .env\n"
                "- **Ollama**: Install from https://ollama.ai"
            )

        # Show current status
        if provider_options:
            current = st.session_state.llm_provider
            if current == "openai":
                st.caption("Using OpenAI API")
            elif current == "ollama":
                st.caption("Using Ollama (local)")

        st.divider()

        # Settings
        with st.expander("⚙️ Settings"):
            st.number_input("Top-K Results", min_value=1, max_value=20, value=6, key="top_k")
            st.checkbox("Show source text", value=True, key="show_sources")
            st.checkbox("Show similarity scores", value=False, key="show_scores")


def chat_interface():
    """Render main chat interface."""
    st.title("💬 RAG Document QA")
    st.caption("Ask questions about your documents")

    # Check if index is built
    if not st.session_state.index_built:
        st.warning(
            "⚠️ No index found. Please upload documents and build the index using the sidebar.",
            icon="⚠️"
        )
        return

    # Initialize chatbot
    if st.session_state.chatbot is None:
        try:
            with st.spinner(f"Loading chatbot with {st.session_state.llm_provider}..."):
                st.session_state.chatbot = GroundedChatbot(
                    provider_name=st.session_state.llm_provider
                )
        except Exception as e:
            st.error(f"Error loading chatbot: {e}")
            st.info("Running in retrieval-only mode (no LLM generation)")
            return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 View Sources"):
                    for i, source_doc in enumerate(message["sources"], 1):
                        metadata = source_doc.metadata
                        st.markdown(f"**{i}. {metadata.get('filename', 'Unknown')}** (page {metadata.get('page', '?')})")

                        if st.session_state.show_sources:
                            st.text_area(
                                "Source text",
                                source_doc.page_content,
                                height=100,
                                key=f"source_{message['msg_id']}_{i}",
                                disabled=True
                            )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.generate_answer(prompt)

                    st.markdown(response["answer"])

                    # Store message with sources
                    msg_id = len(st.session_state.messages)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["source_documents"],
                        "msg_id": msg_id
                    })

                    # Show sources
                    with st.expander("📄 View Sources"):
                        for i, source_doc in enumerate(response["source_documents"], 1):
                            metadata = source_doc.metadata
                            st.markdown(f"**{i}. {metadata.get('filename', 'Unknown')}** (page {metadata.get('page', '?')})")

                            if st.session_state.show_sources:
                                st.text_area(
                                    "Source text",
                                    source_doc.page_content,
                                    height=100,
                                    key=f"source_{msg_id}_{i}",
                                    disabled=True
                                )

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


def main():
    """Main Streamlit app."""
    initialize_session_state()
    sidebar()
    chat_interface()


if __name__ == "__main__":
    main()
