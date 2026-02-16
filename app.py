import streamlit as st
import os
import time
from dotenv import load_dotenv

from src.pdf_loader import load_pdf
from src.vectorstore import (
    create_vectorstore,
    load_vectorstore,
    delete_vectorstore,
    list_vectorstores
)
from src.rag_chain import create_chain
from src.utils import save_uploaded_file

load_dotenv()

st.set_page_config(page_title="Multi-PDF AI Expert", layout="wide")

st.title("📚 Multi-Document AI Knowledge Base")

# -------------------------
# SESSION STATE
# -------------------------

if "chain" not in st.session_state:
    st.session_state.chain = None

if "current_doc" not in st.session_state:
    st.session_state.current_doc = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# SIDEBAR
# -------------------------

with st.sidebar:

    st.header("📂 Upload New PDF")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if st.button("Process Document"):

        if uploaded_file:

            path = save_uploaded_file(uploaded_file)

            docs = load_pdf(path)

            doc_name = uploaded_file.name

            create_vectorstore(docs, doc_name)

            st.success(f"Added: {doc_name}")

# -------------------------
# DOCUMENT SELECTOR
# -------------------------

st.sidebar.header("📚 Knowledge Base")

docs = list_vectorstores()

selected_doc = st.sidebar.selectbox(
    "Select document",
    docs if docs else ["No documents"]
)

# LOAD SELECTED DOC

if selected_doc and selected_doc != "No documents":

    if st.session_state.current_doc != selected_doc:

        vs = load_vectorstore(selected_doc)

        st.session_state.chain = create_chain(vs)

        st.session_state.current_doc = selected_doc

        st.session_state.messages = []

# -------------------------
# DELETE BUTTON
# -------------------------

if selected_doc and selected_doc != "No documents":

    if st.sidebar.button("🗑 Delete Selected Document"):

        delete_vectorstore(selected_doc)

        st.success("Document deleted")

        st.rerun()

# -------------------------
# CHAT HISTORY
# -------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# STREAMING FUNCTION
# -------------------------

def stream_answer(text):

    placeholder = st.empty()

    full = ""

    for char in text:

        full += char

        placeholder.markdown(full + "▌")

        time.sleep(0.003)

    placeholder.markdown(full)

    return full

# -------------------------
# CHAT INPUT
# -------------------------

if prompt := st.chat_input("Ask question..."):

    if not st.session_state.chain:

        st.warning("Select or upload document first")

        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        result = st.session_state.chain.invoke({
            "query": prompt
        })

        answer = stream_answer(result["result"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
