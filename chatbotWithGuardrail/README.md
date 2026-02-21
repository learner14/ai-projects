# Chatbot With Guardrails (FastAPI + LangChain + NeMo Guardrails)

This project exposes a FastAPI endpoint that runs an OpenAI-backed chatbot through **NeMo Guardrails** input safety checks before generating a response.

## What this app does

- Accepts a user query via `GET /chat/?query=...`
- Applies input moderation/validation rules from `config/config.yml`
- Sends allowed prompts to an LLM using LangChain
- Returns a chatbot response

## Tech stack

- FastAPI (`app.py`)
- LangChain + OpenAI (`ChatOpenAI`)
- NeMo Guardrails (`config/config.yml`)
- Docker + Docker Compose

## Project structure

```text
.
├── app.py
├── config/
│   └── config.yml
├── requirements.txt
├── Dockerfile
├── compose.yaml
└── README.md
```

## Guardrails behavior

Guardrails are defined in `config/config.yml`.

Current policy blocks messages that are harmful, abusive, explicit, attempt prompt injection/jailbreak behavior, include sensitive data, or ask for code execution/system prompt leakage.

To customize behavior:

1. Edit `config/config.yml`
2. Update rails flows/prompts as needed
3. Restart the app

## Prerequisites

- Python 3.11+
- pip
- Docker Desktop (for container workflows)
- OpenAI API access

## Local development setup

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Configure credentials (important)

Set credentials as environment variables (recommended):

```bash
export OPENAI_API_KEY="your_openai_api_key"
export ACTIVELOOP_TOKEN="your_activeloop_token"
```

> Note: `app.py` currently includes hardcoded keys. Move secrets to environment variables before sharing or deploying this project.

### 4) Run the API locally

```bash
fastapi dev app.py
```

API will be available at:

- `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

### 5) Test endpoint

```bash
curl "http://127.0.0.1:8000/chat/?query=Write%20a%20short%20introduction%20to%20FastAPI"
```

---

## Docker: build and run

### Build image

```bash
docker build -t chatbot-guardrails .
```

### Run container

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="your_openai_api_key" \
  -e ACTIVELOOP_TOKEN="your_activeloop_token" \
  chatbot-guardrails
```

### Verify

```bash
curl "http://localhost:8000/chat/?query=Hello"
```

---

## Docker Compose

Run with compose:

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

If you need environment variables with Compose, create a `.env` file (same folder as `compose.yaml`) and add:

```bash
OPENAI_API_KEY=your_openai_api_key
ACTIVELOOP_TOKEN=your_activeloop_token
```

Then run:

```bash
docker compose --env-file .env up --build
```

---

## Deployment guide

You can deploy this app to any container platform (Azure Container Apps, AWS ECS/Fargate, Google Cloud Run, Render, Railway, etc.).

### Standard deployment steps

1. Build image:
   ```bash
   docker build -t chatbot-guardrails:latest .
   ```
2. Push image to your container registry
3. Create a service using container port `8000`
4. Set environment variables:
   - `OPENAI_API_KEY`
   - `ACTIVELOOP_TOKEN`
5. Configure health checks (recommended):
   - Path: `/docs` or `/openapi.json`
6. Enable logs/monitoring
7. Scale replicas as needed

### Production recommendations

- Remove hardcoded secrets from source code
- Pin exact package versions in `requirements.txt`
- Add request logging and structured error handling
- Add rate limiting and authentication if this endpoint is public

---

## Troubleshooting

- `ModuleNotFoundError`: run `pip install -r requirements.txt`
- `OPENAI` auth errors: verify `OPENAI_API_KEY`
- Guardrails blocking valid input: adjust prompt policy in `config/config.yml`
- Port conflict on `8000`: run on another port (for Docker, map `-p 8001:8000`)

## Useful commands

```bash
# activate venv
source .venv/bin/activate

# run locally
fastapi dev app.py

# build docker image
docker build -t chatbot-guardrails .

# run with docker
docker run --rm -p 8000:8000 chatbot-guardrails
```



