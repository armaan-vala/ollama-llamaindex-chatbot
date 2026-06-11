# ============================================================
# STEP 2: FastAPI + LlamaIndex Query Engine
# ============================================================
# Ye local API server hai — user question bhejta hai,
# LlamaIndex relevant docs dhundta hai, Ollama jawab deta hai.

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------- LLM aur Embedding setup ----------
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
llm = Ollama(model="llama3", request_timeout=600.0, base_url=ollama_host)
embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embedding

# ---------- Saved index load karo ----------
print("Loading saved index from ./storage/...")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# ---------- Query engine banao ----------
# system prompt se LLM ko batate hain ki tu restaurant assistant hai
query_engine = index.as_query_engine(
    similarity_top_k=3,
    system_prompt=(
        "You are a helpful assistant for Spice Garden Restaurant. "
        "Answer questions based on the restaurant's menu, FAQ, and offers data provided to you. "
        "Be friendly and helpful. If you don't know something, say so honestly. "
        "Always mention prices in ₹ when talking about menu items. "
        "Keep answers concise but complete."
    )
)

print("✓ Query engine ready!")

# ---------- FastAPI app ----------
app = FastAPI(title="Spice Garden AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@app.get("/")
def home():
    return {"status": "running", "restaurant": "Spice Garden AI Assistant"}

@app.post("/ask", response_model=Answer)
def ask_question(q: Question):
    print(f"\nQuestion: {q.question}")
    try:
        response = query_engine.query(q.question)
        print(f"Answer: {response}")
        return Answer(answer=str(response))
    except Exception as e:
        print(f"Error: {e}")
        return Answer(answer="Sorry, model is taking too long. Please try again.")

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 API starting at http://localhost:8000")
    print("📖 Docs at http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
