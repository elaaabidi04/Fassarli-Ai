from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma

def ingest(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    return f"✅ Indexed {len(chunks)} chunks"