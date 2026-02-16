"""
AI PDF Expert - RAG Package

Modules:
- pdf_loader : Load PDF documents
- vectorstore : Create and load Chroma vector database
- rag_chain : Create Retrieval QA chain
- utils : Helper utilities
"""

__version__ = "1.0.0"

# Optional convenient imports
from .pdf_loader import load_pdf
from .vectorstore import create_vectorstore, load_vectorstore
from .rag_chain import create_chain
from .utils import save_uploaded_file
