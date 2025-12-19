from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs


def filter_data_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only source metadata + content."""
    minimal_docs: List[Document] = []

    for doc in docs:
        metadata = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": metadata}
            )
        )

    return minimal_docs


def split_data(minimal_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = splitter.split_documents(minimal_docs)
    return text_chunks


def download_embedding_model():
    """Load HuggingFace embeddings (latest correct import)."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    return embedding
