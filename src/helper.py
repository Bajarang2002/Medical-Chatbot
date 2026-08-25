import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    return loader.load()


def filter_data_to_minimal_docs(
    docs: List[Document]
) -> List[Document]:

    minimal_docs = []

    for doc in docs:
        source = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": source
                }
            )
        )

    return minimal_docs


def split_data(minimal_docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    return splitter.split_documents(
        minimal_docs
    )


def download_embedding_model():

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print(
        "Loading embedding model:",
        model_name,
        flush=True
    )

    embedding = HuggingFaceEmbeddings(
        model_name=model_name
    )

    print(
        "Embedding model loaded successfully.",
        flush=True
    )

    return embedding