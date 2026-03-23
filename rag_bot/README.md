# Advanced RAG Telegram Bot

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot built using **ChromaDB, Sentence Transformers, and Ollama (LLM)**.
This bot can answer questions from custom documents, maintain conversation context, and provide summarized insights.

---

## Features

- **RAG Pipeline** (Retrieval + Generation)
- **ChromaDB Vector Database** for semantic search
- **Message History Awareness** (last 3 interactions per user)
- **Query Caching** for faster responses
- **Source Attribution** (shows document used)
- **/summarize Command** for chat summarization
- **Streaming Responses** (real-time output like ChatGPT)
- **Local LLM using Ollama** (no external API dependency)

---

## System Architecture

```
User Query
   ↓
Telegram Bot
   ↓
Query Routing
   ↓
Embedding (SentenceTransformer)
   ↓
ChromaDB (Vector Search)
   ↓
Top-K Retrieved Chunks
   ↓
Ollama LLM (phi3)
   ↓
Final Answer + Source
```

---

## Tech Stack

- **Language**: Python
- **LLM**: Ollama (phi3)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector DB**: ChromaDB
- **Framework**: python-telegram-bot
- **API**: Ollama Local API (`http://localhost:11434`)

---

## Project Structure

```
project/
│
├── app.py
├── rag.py
├── data/
│   ├── ai.txt
│   ├── ml.txt
│   ├── rag.txt
|
│__ requirements.txt
|
└── README.md

```

---

## How to Run Locally

### 1️⃣ Clone the repository

git clone https://github.com/adithyabarda/rag-telegram-bot
cd project-folder

---

### 2️⃣ Create virtual environment

python -m venv myenv
myenv\Scripts\activate # Windows

---

### 3️⃣ Install dependencies

pip install -r requirements.txt

---

### 4️⃣ Install and start Ollama

Download from: https://ollama.com

Run:

ollama run phi3

---

### 5 Run the bot

python app.py

---

## Bot Commands

| Command           | Description                   |
| ----------------- | ----------------------------- |
| `/start`          | Start the bot                 |
| `/ask <question>` | Ask a question                |
| `/summarize`      | Summarize last 3 interactions |
| `/help`           | Show commands                 |

---

## Models Used

### Embedding Model

- `all-MiniLM-L6-v2`

### LLM (via Ollama)

- `phi3` (fast, lightweight)

---

## How RAG Works in This Project

1. Documents are split into chunks
2. Chunks are converted into embeddings
3. Stored in ChromaDB
4. User query → embedding
5. Top relevant chunks retrieved
6. Sent to LLM with prompt
7. LLM generates final answer

---

## Notes

- Ollama must be running before starting the bot
- Works completely **offline (local LLM)**

---

## Example

/ask What is RAG?
→ Answer + 📄 Source: rag.txt

/ask Who is Sachin?
→ I don’t know.

---
