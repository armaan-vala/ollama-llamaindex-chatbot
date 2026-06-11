# Ollama + LlamaIndex Chatbot

A simple RAG (Retrieval-Augmented Generation) chatbot built with **Ollama**, **LlamaIndex**, and **FastAPI**. This project was created to get hands-on practice with local LLMs and document-based Q&A — nothing fancy, just the basics done right.

The bot answers questions about a fictional restaurant "Spice Garden" using its menu, FAQ, and offers data. Everything runs **100% locally** — no API keys, no cloud, no paid services.

## Tech Stack

| Tool | Role |
|------|------|
| [Ollama](https://ollama.com) | Runs LLaMA 3 locally on your machine |
| [LlamaIndex](https://www.llamaindex.ai) | Loads documents, creates embeddings, retrieves relevant chunks |
| [FastAPI](https://fastapi.tiangolo.com) | Serves the REST API with auto-generated Swagger docs |
| [HuggingFace Embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5) | Converts text to vectors for similarity search |

## How It Works

```
User Question → FastAPI → LlamaIndex (find relevant docs) → Ollama/LLaMA 3 (generate answer) → Response
```

1. Documents (menu, FAQ, offers) are loaded and split into chunks
2. Each chunk is converted to a vector embedding using `bge-small-en-v1.5`
3. When a question comes in, LlamaIndex finds the top 3 most relevant chunks
4. Those chunks + the question are sent to LLaMA 3 via Ollama
5. LLaMA 3 generates a natural language answer

## Project Structure

```
ollama-llamaindex-chatbot/
├── data/
│   ├── menu.txt           # Restaurant menu with prices
│   ├── faq.txt            # Timings, location, delivery info
│   └── offers.txt         # Current offers and combos
├── index_documents.py     # Step 1: Index documents and save embeddings
├── app.py                 # Step 2: FastAPI server with /ask endpoint
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container setup
├── docker-compose.yml     # Run app + Ollama together
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- LLaMA 3 model pulled: `ollama pull llama3`

### Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/ollama-llamaindex-chatbot.git
cd ollama-llamaindex-chatbot

# Install dependencies
pip install -r requirements.txt

# Step 1: Index the documents (one-time)
python index_documents.py

# Step 2: Start the API server
python app.py
```

API will be running at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

### Run with Docker

```bash
docker-compose up --build
```

### Try It Out

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "butter chicken ka price kya hai?"}'
```

Or open `http://localhost:8000/docs` in your browser and use the Swagger UI.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/ask` | Send a question, get an answer |

### Request Body

```json
{
  "question": "koi offer chal raha hai?"
}
```

### Response

```json
{
  "answer": "Yes! There are several offers including Family Feast for ₹1499 (save ₹601), Lunch Express combo for ₹249, and 20% off during Happy Hours..."
}
```

## What I Learned

- How to run LLMs locally using Ollama
- RAG pipeline basics: load → chunk → embed → retrieve → generate
- LlamaIndex document indexing and query engine
- Building APIs with FastAPI
- Connecting all pieces together for a working chatbot

## Note

This is a learning project — intentionally kept simple. The restaurant data is fictional. If you're learning RAG or local LLMs, feel free to fork and swap in your own data.

## License

MIT
