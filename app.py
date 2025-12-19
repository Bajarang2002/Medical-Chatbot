from flask import Flask, request, render_template
from src.helper import download_embedding_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from src.prompts import *
from langchain_core.runnables import RunnablePassthrough
import os

app = Flask(__name__)

# Load API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Embeddings
embeddings = download_embedding_model()

# Pinecone index
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

# Retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("assistant", "Relevant context: {context}")
])

# Model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# NEW RAG PIPELINE (No deprecated chains)
rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | model
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST", "GET"])
def chat():
    user_msg = request.form['msg']
    print("User:", user_msg)

    response = rag_chain.invoke(user_msg)
    final_answer = response.content  # Gemini returns text as content

    print("Response:", final_answer)
    return str(final_answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
