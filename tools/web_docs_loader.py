"""
Web Docs loader – fetches web pages (docs, blogs, tutorials), cleans HTML to text,
and chunks content for embedding and indexing in the vector store.
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_urls(urls: list[str]) -> list[Document]:
    """Fetch URLs, extract text from HTML, return list of Documents with source metadata."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    for d in docs:
        if "source" not in d.metadata:
            d.metadata["source"] = "web_docs"
    return docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_chunk_urls(
    urls: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Load web pages from URLs and return chunked documents ready for the vector store."""
    docs = load_urls(urls)
    return chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
