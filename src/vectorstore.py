import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DB_DIR = "chroma_db"


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )



def get_doc_db_path(doc_name):

    return os.path.join(BASE_DB_DIR, doc_name)


def create_vectorstore(documents, doc_name):

    db_path = get_doc_db_path(doc_name)

    os.makedirs(db_path, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    chunks = [c for c in chunks if c.page_content.strip()]

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    vectorstore.persist()

    return vectorstore


def load_vectorstore(doc_name):

    db_path = get_doc_db_path(doc_name)

    if not os.path.exists(db_path):
        return None

    embeddings = get_embeddings()

    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )


def delete_vectorstore(doc_name):

    db_path = get_doc_db_path(doc_name)

    if os.path.exists(db_path):
        shutil.rmtree(db_path)


def list_vectorstores():

    if not os.path.exists(BASE_DB_DIR):
        return []

    return os.listdir(BASE_DB_DIR)
