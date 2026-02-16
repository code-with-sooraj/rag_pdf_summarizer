import streamlit as st

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


def create_chain(vectorstore):

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        streaming=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return chain
