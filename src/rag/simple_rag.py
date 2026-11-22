"""
RAG module with Streamlit-safe vector store loading.

Uses a module-level singleton pattern to avoid multiple Chroma client
initializations which cause KeyError in Streamlit.
"""

import atexit
from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb


NOTES_DIR = Path(__file__).parent.parent.parent / "data" / "notes"
CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level singletons
_vectorstore: Optional[Chroma] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None
_client: Optional[chromadb.PersistentClient] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create embeddings singleton."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_chroma_client() -> chromadb.PersistentClient:
    """Get or create Chroma persistent client singleton."""
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def build_vectorstore() -> Chroma:
    """Load notes, split, embed, and persist to Chroma."""
    global _vectorstore
    
    documents = []
    for txt_file in NOTES_DIR.glob("*.txt"):
        loader = TextLoader(str(txt_file))
        documents.extend(loader.load())

    if not documents:
        raise ValueError(f"No .txt files found in {NOTES_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = _get_embeddings()
    client = _get_chroma_client()

    # Delete existing collection if it exists to avoid conflicts
    try:
        client.delete_collection("langchain")
    except Exception:
        pass

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="langchain",
    )
    return _vectorstore


def get_vectorstore() -> Chroma:
    """
    Load existing vector store or build if not exists.
    
    Uses singleton pattern to ensure only one Chroma client is created
    per process, which is critical for Streamlit compatibility.
    """
    global _vectorstore
    
    if _vectorstore is not None:
        return _vectorstore
    
    embeddings = _get_embeddings()
    client = _get_chroma_client()
    
    # Check if collection exists and has data
    try:
        collections = client.list_collections()
        has_data = any(c.name == "langchain" and c.count() > 0 for c in collections)
    except Exception:
        has_data = False
    
    if has_data:
        _vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings,
        )
    else:
        _vectorstore = build_vectorstore()
    
    return _vectorstore


def rebuild_vectorstore() -> Chroma:
    """Force rebuild the vector store from source documents."""
    global _vectorstore
    _vectorstore = None
    return build_vectorstore()


def _cleanup():
    """Cleanup function to close Chroma client on exit."""
    global _client, _vectorstore
    if _vectorstore is not None:
        try:
            _vectorstore._client = None
        except Exception:
            pass
    if _client is not None:
        try:
            _client = None
        except Exception:
            pass


atexit.register(_cleanup)


if __name__ == "__main__":
    vs = build_vectorstore()
    results = vs.similarity_search("diabetes treatment", k=2)
    for doc in results:
        print(f"---\n{doc.page_content[:200]}...")