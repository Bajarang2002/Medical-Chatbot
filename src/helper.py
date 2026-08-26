import os
from langchain_community.document_loaders import (PyPDFLoader,DirectoryLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache
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


@lru_cache(maxsize=1)
def download_embedding_model():
    print("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print("Embedding model loaded successfully.")

    return embedding