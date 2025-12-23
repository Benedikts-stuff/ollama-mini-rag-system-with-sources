import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "./db"
FILES_DIR = "./Files"

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_or_update_vector_db():
    print(f"Scanne Dateien in {PERSIST_DIR}...")

    loader = PyPDFDirectoryLoader(FILES_DIR)
    documents = loader.load()

    if not documents:
        return None, f"Keine PDFs im Ordner {FILES_DIR} gefunden."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    print(f"Chunks: {chunks} erstellt.")

    embedding_function = get_embedding_function()

    vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=PERSIST_DIR)

    return vector_db, f"Erfolg: {len(documents)} Dokumente indexiert ({len(chunks)} erstellt)."


def load_vector_db():
    embedding_function = get_embedding_function()

    if os.path.exists(PERSIST_DIR):
        vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_function)
        return vector_db
    else:
        return None


