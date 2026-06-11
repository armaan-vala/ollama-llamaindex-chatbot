# ============================================================
# STEP 1: LlamaIndex se documents index karo
# ============================================================
# Ye script ek baar run hoga — data/ folder ke sab documents
# padhega, chunks me todega, embeddings banayega, aur locally
# save kar dega taaki baar baar process na karna pade.

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------- Ollama se LLaMA 3 connect karo ----------
# Ollama background me chal raha hai localhost:11434 pe
# LlamaIndex usse baat karega jab jawab generate karna hoga
llm = Ollama(model="llama3", request_timeout=120.0)

# ---------- Embedding model setup ----------
# Ye chhota model hai (locally chalega) jo text ko vectors me convert karta hai
# LLaMA 3 se alag hai — ye sirf vectors banata hai, jawab nahi deta
embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embedding

# ---------- Documents load karo ----------
print("Loading documents from data/ folder...")
documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded {len(documents)} documents")

for doc in documents:
    print(f"  → {doc.metadata.get('file_name', 'unknown')} ({len(doc.text)} chars)")

# ---------- Index banao ----------
# Ye step:
# 1. Documents ko chhote chunks me todta hai
# 2. Har chunk ka embedding (vector) banata hai
# 3. Vectors ko memory me store karta hai
print("\nIndexing documents (embedding generation)...")
print("(Pehli baar me embedding model download hoga ~130MB, ek baar hi hoga)")

index = VectorStoreIndex.from_documents(documents)

# ---------- Index save karo locally ----------
# Taaki baar baar index banana na pade
index.storage_context.persist(persist_dir="./storage")
print("\n✓ Index saved to ./storage/")
print("→ Step 1 complete! Ab app.py run karke API start karo.")
