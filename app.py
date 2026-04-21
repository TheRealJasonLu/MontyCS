import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

DB_PATH = "vectordb"

st.title("MontyCS AI Assistant")

# load vector DB
embedding = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model="mistral")

query = st.text_input("Ask a question:")

if query:
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    st.write("### Answer")
    st.write(response)

    st.write("### Sources")
    for doc in docs:
        st.write("-", doc.metadata.get("source"))