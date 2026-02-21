# Semantic Search in PDF using Agentic AI

FastAPI application that parses a PDF, creates embeddings, stores chunks in a Deep Lake vector database, and answers user queries with retrieval-augmented generation (RAG).

## Features

- Parse PDF files with `PyPDFLoader`
- Split text into chunks for embedding
- Store vectors in Deep Lake (`./my_deeplake`)
- Run semantic similarity search on query
- Generate answer with OpenAI chat model
- Expose API via FastAPI (`/chat/`)

## Project Structure

- `app.py` - FastAPI app and semantic search flow
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container image definition
- `compose.yaml` - Docker Compose local deployment
- `my_deeplake/` - Local Deep Lake dataset directory

## Prerequisites

- Python 3.11+
- OpenAI API key
- ActiveLoop token
- Docker and Docker Compose (for containerized run)

## Environment Variables

Set these before running the app:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export ACTIVELOOP_TOKEN="your_activeloop_token"
```

Note: `app.py` currently contains hardcoded placeholder values for both keys. Replace those values or remove those lines so environment variables are used securely.

## Local Development

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

### 3) Run FastAPI app

```bash
fastapi dev app.py
```

or with uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

App runs at: `http://localhost:8000`

## API Usage

### Semantic Search Endpoint

`GET /chat/?query=<your_question>`

Example:

```bash
curl "http://localhost:8000/chat/?query=What does this PDF say about Linux commands?"
```

## Build

### Docker image build

```bash
docker build -t pdf-semantic-search:latest .
```

### Run container

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e ACTIVELOOP_TOKEN="$ACTIVELOOP_TOKEN" \
  pdf-semantic-search:latest
```

## Deployment

## Option 1: Docker Compose (local/staging)

```bash
docker compose up --build
```

Service URL: `http://localhost:8000`

## Option 2: Cloud container deployment

1. Build image:

```bash
docker build -t <registry>/<image>:<tag> .
```

2. For cross-platform build (Apple Silicon to amd64 cloud):

```bash
docker build --platform=linux/amd64 -t <registry>/<image>:<tag> .
```

3. Push image:

```bash
docker push <registry>/<image>:<tag>
```

4. Deploy to your cloud runtime (ECS, Cloud Run, AKS, etc.) with:
   - Port `8000`
   - Environment variables `OPENAI_API_KEY` and `ACTIVELOOP_TOKEN`
   - Persistent storage if you want to retain local vector data across restarts

## Testing

This repository currently has no automated test suite. Use the following functional tests.

### 1) Health/basic startup test

```bash
curl "http://localhost:8000/docs"
```

Expected: Swagger UI HTML response.

### 2) Semantic search smoke test

```bash
curl "http://localhost:8000/chat/?query=Summarize the PDF"
```

Expected: model-generated answer using retrieved PDF context.

### 3) Container smoke test

```bash
docker compose up --build
curl "http://localhost:8000/chat/?query=What topics are in this document?"
```

## Known Limitations

- PDF file name is currently fixed in `app.py` (`The One Page Linux Manual.pdf`)
- Vector store is rebuilt on each request (`overwrite=True`)
- API keys should not be hardcoded in source

## Next Improvements

- Add upload endpoint for dynamic PDF ingestion
- Persist and reuse vector index across requests
- Add `pytest` tests for API and retrieval flow
- Add CI pipeline for linting and tests



