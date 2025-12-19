
from src.helper import load_pdf_file, filter_data_to_minimal_docs, split_data, download_embedding_model
from typing import List
from langchain.schema import Documents
from dotenv  import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()


extracted_data =load_pdf_file("Data")
minimal_docs = filter_data_to_minimal_docs(extracted_data)
text_chunks = split_data(minimal_docs)
embeddings = download_embedding_model()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)




index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # the number of dimensions in your embeddings (e.g. 768)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-2")
    )

index = pc.Index(index_name)



vectorstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding= embeddings,
)
    









