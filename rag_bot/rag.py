from sentence_transformers import SentenceTransformer
import chromadb
import requests

# =========================
# 1. MODEL
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# 2. CHROMA DB
# =========================
client = chromadb.Client()
collection = client.get_or_create_collection(name="rag_collection")

# =========================
# 3. CHUNKING
# =========================
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# =========================
# 4. LOAD + STORE WITH METADATA
# =========================
docs = []
ids = []
metadatas = []

file_names = ["ai.txt", "ml.txt", "rag.txt"]

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

print(f"Stored {len(docs)} chunks in ChromaDB")

# =========================
# 5. CACHE
# =========================
query_cache = {}

# =========================
# 6. RETRIEVE (WITH SOURCE)
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
# 7. GENERATE (CLEAN OUTPUT)
# =========================
def stream_answer(query, retrieved_data, history=""):
    docs = [item[0] for item in retrieved_data]
    sources = [item[1]["source"] for item in retrieved_data]

    context = " ".join(docs)

    # 🔥 OPTIONAL: basic relevance check (very useful)
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
                json_data = json.loads(line.decode("utf-8"))

                token = json_data.get("response", "")
                full_answer += token

                yield full_answer   # streaming

            except:
                continue

    clean_answer = full_answer.strip().lower()

    # 🔥 IMPORTANT: do NOT show source for fallback answers
    if "i don't know" in clean_answer or "not sure" in clean_answer:
        yield full_answer.strip()
        return

    # ✅ show sources only if valid answer
    unique_sources = list(set(sources))
    yield f"{full_answer.strip()}\n\n📄 Source: {', '.join(unique_sources)}"