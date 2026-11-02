# Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-PostgreSQL

## 🚀 Live Demo

**Deployed Application:**  
https://medical-chatbot-11-iqwl.onrender.com/

---

# How to run?

### STEPS:

Clone the repository

```bash
git clone https://github.com/Bajarang2002/Medical-Chatbot.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

Activate the environment

```bash
conda activate medibot
```

### STEP 02- Install the requirements

```bash
pip install -r requirements.txt
```

### STEP 03- Create a `.env` file in the root directory

Add the following environment variables:

```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
DATABASE_URL=your_postgresql_database_url
```

### STEP 04- Run the application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:8000
```

---

# 🏗️ Project Architecture

```text
User Question
      │
      ▼
Flask Application
      │
      ▼
Conversation History
      │
      ▼
PostgreSQL Database
      │
      ▼
Embedding Model
sentence-transformers/all-MiniLM-L6-v2
      │
      ▼
Pinecone Vector Database
      │
      ▼
Relevant Medical Documents
      │
      ▼
LangChain RAG Pipeline
      │
      ▼
Google Gemini
      │
      ▼
AI Generated Response
      │
      ▼
User
```

---

# 🛠️ Tech Stack Used

- Python
- Flask
- LangChain
- Google Gemini
- Pinecone
- Sentence Transformers
- PostgreSQL
- Psycopg2
- Retrieval-Augmented Generation (RAG)
- Render

---

# ✨ Features

- AI-powered Medical Chatbot
- Retrieval-Augmented Generation (RAG)
- Medical document retrieval using Pinecone
- Google Gemini integration
- Conversation history management
- PostgreSQL database integration
- Create multiple conversations
- Automatic conversation title generation
- Delete conversations
- Streaming AI responses
- Secure environment variable management

---

# 📂 Project Structure

```text
Medical-Chatbot/
│
├── src/
│   ├── helper.py
│   └── prompts.py
│
├── templates/
│   └── chat.html
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🧠 How It Works

### STEP 01- User asks a medical question

The user enters a question through the Medical Chatbot interface.

### STEP 02- Conversation history is retrieved

Previous conversation messages are stored and retrieved from the PostgreSQL database.

### STEP 03- Search relevant medical information

The user's question is converted into embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The application searches the Pinecone Vector Database for relevant medical documents.

### STEP 04- Create RAG context

The retrieved medical documents, conversation history, and current user question are combined into a prompt.

### STEP 05- Generate AI response

The prompt is sent to Google Gemini through LangChain.

### STEP 06- Stream and save the response

The generated AI response is streamed back to the user and saved in the PostgreSQL database.

---

# 🔗 API Endpoints

### Home Page

```http
GET /
```

### Create New Chat

```http
POST /new_chat
```

### Get All Conversations

```http
GET /conversations
```

### Get Conversation Messages

```http
GET /conversation/<conversation_id>
```

### Delete Conversation

```http
DELETE /conversation/<conversation_id>
```

### Chat with AI

```http
POST /chat
```

Request parameters:

```text
msg
conversation_id
```

---

# 🗄️ Database Structure

### Conversations Table

```text
conversations
├── id
├── title
├── created_at
└── updated_at
```

### Messages Table

```text
messages
├── id
├── conversation_id
├── role
├── content
└── created_at
```

---

# 🌐 Deployment

The application is deployed on **Render**.

## Live Application

https://medical-chatbot-11-iqwl.onrender.com/

Make sure the following environment variables are configured:

```text
PINECONE_API_KEY
GOOGLE_API_KEY
DATABASE_URL
```

---

# 🔒 Important

Do not upload your `.env` file to GitHub.

Add the following to `.gitignore`:

```text
.env
venv/
__pycache__/
*.pyc
```

---

# ⚠️ Medical Disclaimer

This Medical Chatbot is designed for **educational and informational purposes only**.

It should not be considered a replacement for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

---

# 👨‍💻 Author

**Bajarang Dhamanekar**

AI/ML Engineer | Generative AI Enthusiast

---

# ⭐ Support

If you like this project, please give it a **Star ⭐** on GitHub!
