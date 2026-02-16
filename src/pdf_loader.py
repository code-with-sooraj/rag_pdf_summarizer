from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):

    loader = PyPDFLoader(file_path)

    docs = loader.load()

    docs = [doc for doc in docs if doc.page_content.strip()]

    return docs
