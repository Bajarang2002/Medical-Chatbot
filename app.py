from flask import Flask,request,render_template,Response,jsonify
from src.helper import download_embedding_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.prompts import system_prompt
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import uuid
import re
import time
from datetime import datetime

app=Flask(__name__)

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
DATABASE_URL=os.getenv("DATABASE_URL")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set.")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set.")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set.")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY


def get_db():
    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor
    )


def init_db():
    conn=get_db()
    cur=conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations(
            id UUID PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id SERIAL PRIMARY KEY,
            conversation_id UUID NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY(conversation_id)
            REFERENCES conversations(id)
            ON DELETE CASCADE
        )
    """)

    conn.commit()
    cur.close()
    conn.close()


init_db()


print("Loading embeddings...")
embeddings=download_embedding_model()
print("Embeddings loaded.")


index_name="medical-chatbot"

docsearch=PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

print("Pinecone connected.")


retriever=docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)


model=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


def create_title(text):
    text=text.strip()

    if not text:
        return "New Chat"

    text=re.sub(r"\s+"," ",text)
    text=text.replace("\n"," ").strip()

    if len(text)>45:
        text=text[:45].rstrip()+"..."

    return text


def create_conversation():
    conversation_id=str(uuid.uuid4())
    now=datetime.now()

    conn=get_db()
    cur=conn.cursor()

    cur.execute(
        """
        INSERT INTO conversations
        (id,title,created_at,updated_at)
        VALUES(%s,%s,%s,%s)
        """,
        (
            conversation_id,
            "New Chat",
            now,
            now
        )
    )

    conn.commit()
    cur.close()
    conn.close()

    return conversation_id


def save_message(conversation_id,role,content):
    now=datetime.now()

    conn=get_db()
    cur=conn.cursor()

    cur.execute(
        """
        INSERT INTO messages
        (conversation_id,role,content,created_at)
        VALUES(%s,%s,%s,%s)
        """,
        (
            conversation_id,
            role,
            content,
            now
        )
    )

    cur.execute(
        """
        UPDATE conversations
        SET updated_at=%s
        WHERE id=%s
        """,
        (
            now,
            conversation_id
        )
    )

    conn.commit()
    cur.close()
    conn.close()


def get_messages(conversation_id):
    conn=get_db()
    cur=conn.cursor()

    cur.execute(
        """
        SELECT role,content
        FROM messages
        WHERE conversation_id=%s
        ORDER BY id ASC
        """,
        (conversation_id,)
    )

    rows=cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "role":row["role"],
            "content":row["content"]
        }
        for row in rows
    ]


def get_conversations():
    conn=get_db()
    cur=conn.cursor()

    cur.execute(
        """
        SELECT id,title,created_at,updated_at
        FROM conversations
        ORDER BY updated_at DESC
        """
    )

    rows=cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "id":str(row["id"]),
            "title":row["title"],
            "created_at":row["created_at"].isoformat(),
            "updated_at":row["updated_at"].isoformat()
        }
        for row in rows
    ]


def update_title(conversation_id,title):
    conn=get_db()
    cur=conn.cursor()

    cur.execute(
        """
        UPDATE conversations
        SET title=%s
        WHERE id=%s
        """,
        (
            title,
            conversation_id
        )
    )

    conn.commit()
    cur.close()
    conn.close()


def delete_conversation(conversation_id):
    conn=get_db()
    cur=conn.cursor()

    cur.execute(
        """
        DELETE FROM conversations
        WHERE id=%s
        """,
        (conversation_id,)
    )

    conn.commit()
    cur.close()
    conn.close()


def format_docs(docs):
    if not docs:
        return "No relevant medical information was found."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


rag_prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        system_prompt
    ),
    (
        "human",
        """
Conversation History:

{history}

Current Question:

{input}

Medical Context:

{context}

Use the medical context when answering the current question.

Use conversation history to understand follow-up questions.

If the answer cannot be supported by the medical context, clearly say that the information is not available in the provided medical documents.

Do not invent medical facts.
"""
    )
])


@app.route("/")
def index():
    conversations=get_conversations()

    return render_template(
        "chat.html",
        conversations=conversations
    )


@app.route("/new_chat",methods=["POST"])
def new_chat():
    conversation_id=create_conversation()

    return jsonify({
        "success":True,
        "conversation_id":conversation_id,
        "title":"New Chat"
    })


@app.route("/conversations",methods=["GET"])
def conversations():
    return jsonify(
        get_conversations()
    )


@app.route(
    "/conversation/<conversation_id>",
    methods=["GET"]
)
def get_conversation(conversation_id):
    messages=get_messages(conversation_id)

    return jsonify({
        "success":True,
        "messages":messages
    })


@app.route(
    "/conversation/<conversation_id>",
    methods=["DELETE"]
)
def delete_chat(conversation_id):
    delete_conversation(conversation_id)

    return jsonify({
        "success":True
    })


@app.route("/chat",methods=["POST"])
def chat():
    user_msg=request.form.get(
        "msg",
        ""
    ).strip()

    conversation_id=request.form.get(
        "conversation_id",
        ""
    ).strip()

    print("\n==============================")
    print("USER:",user_msg)
    print("THREAD:",conversation_id)
    print("==============================")

    if not user_msg:
        return Response(
            "Please enter a question.",
            mimetype="text/plain"
        )

    if not conversation_id:
        conversation_id=create_conversation()

    conn=get_db()
    cur=conn.cursor()

    cur.execute(
        """
        SELECT id,title
        FROM conversations
        WHERE id=%s
        """,
        (conversation_id,)
    )

    conversation=cur.fetchone()

    cur.close()
    conn.close()

    if not conversation:
        conversation_id=create_conversation()

    existing_messages=get_messages(
        conversation_id
    )

    is_first_message=len(
        existing_messages
    )==0

    if is_first_message:
        title=create_title(user_msg)

        update_title(
            conversation_id,
            title
        )

    save_message(
        conversation_id,
        "user",
        user_msg
    )

    history_messages=get_messages(
        conversation_id
    )

    previous_messages=history_messages[:-1]

    history_text=""

    for message in previous_messages:
        role=message["role"].upper()

        history_text+=(
            f"{role}: "
            f"{message['content']}\n"
        )

    if not history_text:
        history_text="No previous conversation."

    print("\nSearching Pinecone...")

    try:
        docs=retriever.invoke(
            user_msg
        )

        print(
            "Retrieved documents:",
            len(docs)
        )

        for i,doc in enumerate(docs):
            print(
                f"\nDocument {i+1}:"
            )

            print(
                doc.page_content[:500]
            )

        context=format_docs(
            docs
        )

    except Exception as e:
        print(
            "Pinecone retrieval error:",
            str(e)
        )

        context=(
            "No relevant medical "
            "information was found."
        )

    print(
        "\nContext length:",
        len(context)
    )


    def generate():
        full_response=""

        try:
            chain=(
                {
                    "context":lambda x:context,
                    "input":RunnablePassthrough(),
                    "history":lambda x:history_text
                }
                |rag_prompt
                |model
                |StrOutputParser()
            )

            print(
                "\nGenerating response..."
            )

            for chunk in chain.stream(
                user_msg
            ):
                if chunk:
                    full_response+=chunk

                    print(
                        chunk,
                        end="",
                        flush=True
                    )

                    yield chunk

                    time.sleep(0.03)

            save_message(
                conversation_id,
                "assistant",
                full_response
            )

            print(
                "\n\nResponse completed."
            )

        except Exception as e:
            print(
                "\nERROR:",
                str(e)
            )

            error_message=(
                "Sorry, an error occurred: "
                +str(e)
            )

            yield error_message


    return Response(
        generate(),
        mimetype="text/plain; charset=utf-8",
        headers={
            "Cache-Control":"no-cache",
            "X-Accel-Buffering":"no",
            "Connection":"keep-alive"
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )