import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

DB_PATH = "vectordb"

st.set_page_config(page_title="MontyCS AI Assistant", page_icon="💻", layout="centered")

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

.title {
    font-size: 3rem;
    font-weight: 800;
    color: #1f2937;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.05rem;
    color: #6b7280;
    margin-bottom: 2rem;
}

.answer-box {
    background: white;
    padding: 1.4rem 1.4rem 1.1rem 1.4rem;
    border-radius: 18px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
    border: 1px solid #e5e7eb;
    margin-top: 1rem;
}

.answer-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.8rem;
}

.footer {
    margin-top: 2rem;
    color: #9ca3af;
    font-size: 0.95rem;
    text-align: center;
}

.stTextInput > div > div > input {
    border-radius: 14px;
    padding: 0.75rem 1rem;
    font-size: 1.05rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">MontyCS AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions about the Montgomery Computer Science Club.</div>',
    unsafe_allow_html=True
)

embedding = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})
llm = Ollama(model="mistral")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    st.markdown('<div class="answer-title">Answer</div>', unsafe_allow_html=True)
    st.write(response)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Built locally with Ollama, Chroma, and Streamlit.</div>', unsafe_allow_html=True)