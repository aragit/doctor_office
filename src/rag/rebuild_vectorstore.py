#!/usr/bin/env python
"""
Rebuild Chroma vector store utility.

Run this if the vector store becomes corrupted or you need to refresh it:
    python -m src.rag.rebuild_vectorstore
"""

from rag.simple_rag import rebuild_vectorstore


if __name__ == "__main__":
    print("Rebuilding vector store...")
    vs = rebuild_vectorstore()
    print("Vector store rebuilt successfully!")
    
    # Test search
    results = vs.similarity_search("diabetes treatment", k=2)
    for doc in results:
        print(f"---\n{doc.page_content[:200]}...")