from sentence_transformers import SentenceTransformer
import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# 1. MODEL
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# 2. CHROMA DB (PERSISTENT)
# =========================
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="rag_collection")

# =========================
# 3. LANGCHAIN CHUNKING
# =========================
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

# =========================
# 4. LOAD + STORE (ONLY ONCE)
# =========================
file_names = ["ai.txt", "ml.txt", "rag.txt"]

if collection.count() == 0:
    docs = []
    ids = []
    metadatas = []

    id_counter = 0

    for file in file_names:
        with open(f"data/{file}", "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text)

            for chunk in chunks:
                docs.append(chunk)
                ids.append(f"id_{id_counter}")
                metadatas.append({"source": file})
                id_counter += 1

    embeddings = model.encode(docs).tolist()

    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print(f" Stored {len(docs)} chunks in ChromaDB")

else:
    print(f" Loaded existing ChromaDB with {collection.count()} chunks")

# =========================
# 5. CACHE
# =========================
query_cache = {}

# =========================
# 6. RETRIEVE
# =========================
def retrieve(query):
    if query in query_cache:
        return query_cache[query]

    q_embed = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=q_embed,
        n_results=3
    )

    docs = results["documents"][0]
    sources = results["metadatas"][0]

    result = list(zip(docs, sources))

    query_cache[query] = result
    return result

# =========================
# 7. STREAM ANSWER (OLLAMA)
# =========================
def stream_answer(query, retrieved_data, history=""):
    docs = [item[0] for item in retrieved_data]
    sources = [item[1]["source"] for item in retrieved_data]

    context = " ".join(docs)

    # Relevance check
    if len(context.strip()) < 50:
        yield "I don't know."
        return

    prompt = f"""
You are an AI assistant.

Answer ONLY the question asked.
Do NOT include unrelated information.
If the context is not relevant, say "I don't know".

Context:
{context}

Conversation History:
{history}

Question:
{query}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:latest",
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )

        full_answer = ""

        for line in response.iter_lines():
            if line:
                try:
                    import json
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("response", "")
                    full_answer += token

                    yield full_answer

                except:
                    continue

        clean_answer = full_answer.strip().lower()

        # ONLY block source for fallback answers
        if "i don't know" in clean_answer or "not sure" in clean_answer:
            yield full_answer.strip()
            return

       
        unique_sources = list(set(sources))
        yield f"{full_answer.strip()}\n\n📄 Source: {', '.join(unique_sources)}"

    except Exception as e:
        yield f" Error: {str(e)}"
