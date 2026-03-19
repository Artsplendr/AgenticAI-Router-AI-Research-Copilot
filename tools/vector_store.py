"""
Vector store – FAISS-based vector store for RAG.
Uses separate indices per source (paper, web_docs, notes) so each agent has its own retriever.
"""

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


def get_embedding_model() -> Embeddings:
    """Create OpenAI embeddings; uses OPENAI_API_KEY from env."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings()


def _base_path() -> str:
    return os.getenv("VECTOR_DB_PATH", "./data/faiss_index")


def _source_path(source: str) -> str:
    return str(Path(_base_path()) / source)


def get_or_create_vector_store(
    source: str,
    embedding: Embeddings | None = None,
) -> VectorStore:
    """Load or create FAISS index for the given source (paper | web_docs | notes)."""
    path = _source_path(source)
    embedding = embedding or get_embedding_model()
    path_obj = Path(path)
    if path_obj.exists() and (path_obj / "index.faiss").exists():
        return FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)
    # Empty store: create with one dummy doc so FAISS can be built
    dummy = [Document(page_content=" ", metadata={"source": source})]
    return FAISS.from_documents(dummy, embedding)


def add_documents(
    store: VectorStore,
    documents: list[Document],
    source_tag: str,
) -> None:
    """Add documents; source_tag used for metadata only (store is already source-specific)."""
    for doc in documents:
        if not doc.metadata.get("source"):
            doc.metadata["source"] = source_tag
    store.add_documents(documents)


def get_retriever(
    store: VectorStore,
    k: int = 4,
) -> BaseRetriever:
    """Return a similarity retriever for the given store."""
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def save_vector_store(store: VectorStore, source: str) -> None:
    """Persist FAISS store for the given source."""
    path = _source_path(source)
    Path(path).mkdir(parents=True, exist_ok=True)
    if isinstance(store, FAISS):
        store.save_local(path)
