"""
Ingest local notes (.md/.txt) into the FAISS vector store (notes source).
Usage: python scripts/ingest_notes.py [notes_dir]
Default notes_dir: data/notes
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools.vector_store import add_documents, get_or_create_vector_store, save_vector_store


def main():
    load_dotenv(ROOT / ".env")
    notes_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (ROOT / "data/notes")

    if not notes_dir.exists():
        print(f"Notes directory not found: {notes_dir}")
        print("Create it and add .md/.txt files, then rerun.")
        sys.exit(1)

    files = list(notes_dir.glob("*.md")) + list(notes_dir.glob("*.txt"))
    if not files:
        print(f"No note files found in: {notes_dir}")
        print("Add at least one .md or .txt file and rerun.")
        sys.exit(1)

    docs = [
        Document(
            page_content=p.read_text(encoding="utf-8"),
            metadata={"title": p.name, "source": str(p)},
        )
        for p in files
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    store = get_or_create_vector_store("notes")
    add_documents(store, chunks, "notes")
    save_vector_store(store, "notes")
    print(f"Saved notes index with {len(chunks)} chunks from {len(files)} file(s).")


if __name__ == "__main__":
    main()
