import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data/docs"
DB_PATH = "vectordb"

def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())
    return docs

def main():
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embedding = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory=DB_PATH
    )

    db.persist()
    print("Database built successfully.")

if __name__ == "__main__":
    main()