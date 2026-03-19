"""
Ingest web docs from URLs into the FAISS vector store (web_docs source).
Usage: python scripts/ingest_web_docs.py [url1] [url2] ...
Or set WEB_DOCS_URLS in .env (comma-separated).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from tools.web_docs_loader import load_and_chunk_urls
from tools.vector_store import get_or_create_vector_store, add_documents, save_vector_store


def main():
    if len(sys.argv) > 1:
        urls = [u.strip() for u in sys.argv[1:] if u.strip()]
    else:
        import os
        urls_str = os.getenv("WEB_DOCS_URLS", "")
        urls = [u.strip() for u in urls_str.split(",") if u.strip()]
    if not urls:
        print("Usage: python scripts/ingest_web_docs.py <url1> [url2] ...")
        print("Or set WEB_DOCS_URLS=url1,url2 in .env")
        sys.exit(1)
    print("Loading and chunking:", urls)
    docs = load_and_chunk_urls(urls)
    print(f"Got {len(docs)} chunks.")
    store = get_or_create_vector_store("web_docs")
    add_documents(store, docs, "web_docs")
    save_vector_store(store, "web_docs")
    print("Saved web_docs index.")


if __name__ == "__main__":
    main()
