from embeddings import Embedder
from vectorstore import FAISSStore
from groq import Groq

vector_store = FAISSStore(dim=384)  # MiniLM dim
llm = Groq()

def ingest_documents(chunks):
    texts = [c.page_content for c in chunks]
    embeddings = Embedder.embed(texts)
    vector_store.add(embeddings, texts)

def answer_question(question: str) -> str:
    query_embedding = Embedder.embed([question])[0]
    docs = vector_store.search(query_embedding)

    context = "\n".join(docs)

    prompt = f"""
You are a medical assistant.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    completion = llm.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content
