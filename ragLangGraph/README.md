# Agentic RAG API with LangChain + LangGraph

This project is a **Retrieval-Augmented Generation (RAG)** API built with:

- **FastAPI** for serving requests
- **LangChain** for LLM + retrieval orchestration
- **LangGraph** for agent-style tool-calling flow
- **Deep Lake** as the vector store
- **OpenAI** for embeddings and chat completion

The API exposes a single endpoint:

- `GET /chat/?query=<your-question>`

---

## 1) What this app does

At startup, the app:

1. Loads content from a web article
2. Splits it into chunks
3. Stores embeddings in Deep Lake
4. Builds a LangGraph workflow that:
   - decides whether to call a retrieval tool
   - fetches relevant context
   - generates a concise answer

---

## 2) Prerequisites

- Python **3.10+**
- `pip`
- Docker (optional, for container build/deploy)
- OpenAI API key
- ActiveLoop/Deep Lake token

---

## 3) Project structure

```text
ragLangGraph/
├── app.py
├── requirements.txt
├── README.md
└── my_deeplake/
```

---

## 4) Local build and run

### Step A: Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step B: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step C: Configure environment variables

Set secrets as environment variables before running:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export ACTIVELOOP_TOKEN="your_activeloop_token"
```

### Step D: Start the API

```bash
fastapi dev app.py
```

The server should be available at:

- `http://127.0.0.1:8000`

API docs:

- Swagger UI: `http://127.0.0.1:8000/docs`

---

## 5) Test the RAG agent

### Quick browser test

Open:

```text
http://127.0.0.1:8000/chat/?query=What%20is%20Task%20Decomposition?
```

### CLI test with curl

```bash
curl "http://127.0.0.1:8000/chat/?query=What%20is%20Task%20Decomposition%3F"
```

### Smoke-test checklist

- Server starts without import/runtime errors
- `/docs` loads
- `/chat/` returns HTTP 200
- Response is grounded in retrieved context (not generic)

---

## 6) Build Docker image

If your repo includes a `Dockerfile`, build and run with:

```bash
docker build -t rag-langgraph-agent .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e ACTIVELOOP_TOKEN="$ACTIVELOOP_TOKEN" \
  rag-langgraph-agent
```

Then test:

```bash
curl "http://127.0.0.1:8000/chat/?query=Explain%20agentic%20RAG"
```

---

## 7) Deployment options

You can deploy this FastAPI app to:

- Azure Container Apps
- AWS ECS/Fargate
- Google Cloud Run
- Render / Railway / Fly.io

### Generic deployment flow

1. Build image (`docker build`)
2. Push to container registry
3. Create service from image
4. Expose port `8000`
5. Set env vars:
   - `OPENAI_API_KEY`
   - `ACTIVELOOP_TOKEN`
6. Configure health checks and logs

---

## 8) Production recommendations

- Do not hardcode API keys in source code
- Move secrets to environment variables or secret manager
- Add request/response logging and tracing
- Add retries/timeouts for external API calls
- Add automated tests (unit + API integration)
- Pin dependency versions in `requirements.txt`

---

## 9) Troubleshooting

### `ModuleNotFoundError`

```bash
pip install -r requirements.txt
```

### `401` from OpenAI/Deep Lake

Check your environment variables:

```bash
echo $OPENAI_API_KEY
echo $ACTIVELOOP_TOKEN
```

### Port already in use

Run FastAPI on another port:

```bash
fastapi dev app.py --port 8001
```

---

## 10) Example request

```bash
curl "http://127.0.0.1:8000/chat/?query=How%20does%20tool-calling%20work%20in%20this%20agent%3F"
```

---

## 11) Tech stack summary

- FastAPI
- LangChain
- LangGraph
- Deep Lake Vector Store
- OpenAI (LLM + embeddings)


