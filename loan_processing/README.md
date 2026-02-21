# Loan Processing Agentic AI App (crewAI)

This project implements a **multi-agent loan processing system** using the **crewAI** framework and a **hierarchical Supervisor/Orchestrator architecture**.

At runtime, a Manager agent (Supervisor/Orchestrator) coordinates specialized agents for validation, credit checks, risk scoring, and compliance, then compiles a final decision report.

---

## Architecture: Supervisor / Orchestrator Pattern

The crew uses `Process.hierarchical` with a dedicated manager agent:

- **Supervisor/Orchestrator (Manager Agent)**
  - Role: `Loan Processing Manager`
  - Responsibility: delegate tasks, enforce execution order, compile final report/decision.

- **Specialized Worker Agents**
  - `Document Validation Specialist` (`doc_specialist`)
  - `Credit Check Agent` (`credit_analyst`)
  - `Risk Assessment Analyst` (`risk_assessor`)
  - `Compliance Officer` (`compliance_officer`)

Core implementation files:

- `src/loan_processing/crew.py` (agent, task, tools, hierarchical crew config)
- `src/loan_processing/config/agents.yaml` (agent roles/goals)
- `src/loan_processing/config/tasks.yaml` (task pipeline and dependencies)
- `src/loan_processing/main.py` (entrypoints, input prep, run/test helpers)

---

## Workflow Process

The end-to-end workflow is split into distinct stages:

1. **Document Fetch (Preprocessing)**
	- The loan document content is fetched before validation (currently simulated in `main.py` via `get_document_content(document_id)`).

2. **Document Validation**
	- Agent: **Document Validation Specialist**
	- Tool: `ValidateDocumentFieldsTool`
	- Objective: ensure required fields exist and JSON is valid.

3. **Credit Check**
	- Agent: **Credit Check Agent**
	- Tool: `QueryCreditBureauAPITool`
	- Objective: retrieve borrower credit score using `customer_id`.

4. **Risk Assessment**
	- Agent: **Risk Assessment Analyst**
	- Tool: `CalculateRiskScoreTool`
	- Objective: calculate risk using loan amount, income, and credit score.

5. **Compliance Check**
	- Agent: **Compliance Officer**
	- Tool: `CheckLendingComplianceTool`
	- Objective: verify lending-policy and regulatory compliance.

6. **Final Decision Report**
	- Agent: **Loan Processing Manager**
	- Objective: aggregate all outputs and produce final Markdown report.

---

## Prerequisites

- Python `>=3.10,<3.14`
- `uv` (recommended) or `pip`
- OpenAI-compatible model access

Required environment variables:

```bash
export OPENAI_API_KEY="<your_api_key>"
export MODEL="gpt-4o"   # optional, defaults to gpt-4o
```

---

## Setup

### Option A: Using `uv` (recommended)

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Option B: Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Run

From project root:

```bash
crewai run
```

Alternative script entrypoint:

```bash
python -m loan_processing.main
```

Expected behavior:

- Executes hierarchical multi-agent loan workflow.
- Writes final output to `report.md`.

---

## Build

Build a distributable package:

```bash
python -m pip install build
python -m build
```

Artifacts are generated under `dist/`:

- wheel (`.whl`)
- source distribution (`.tar.gz`)

---

## Test

### 1) CrewAI evaluation mode

Use CrewAI's test command:

```bash
crewai test -n 3 -m gpt-4o
```

### 2) Python tests (if/when tests are added under `tests/`)

```bash
pytest -q
```

---

## Deploy

This app is usually deployed as a **containerized worker/API job** that runs the crew on demand.

### Minimal deployment checklist

1. Package app (`python -m build`) or containerize it.
2. Configure runtime env vars:
	- `OPENAI_API_KEY`
	- `MODEL` (optional)
3. Ensure network egress to LLM provider endpoints.
4. Set logs/monitoring (stdout or APM).
5. Trigger execution with valid loan payload/document source.

### Example runtime command in deployment

```bash
crewai run
```

For AWS deployment, common targets include ECS/Fargate, App Runner, or Lambda (with adapted runtime flow).

---

## Project Structure

```text
src/loan_processing/
  crew.py                 # Hierarchical crew + agent tools
  main.py                 # Entrypoints and mock document fetch helper
  config/
	 agents.yaml           # Agent roles and goals
	 tasks.yaml            # Task definitions and dependencies
knowledge/
tests/
report.md
```

---

## Notes

- Current document fetch and credit bureau interactions are mocked/simulated for development.
- Replace mock logic with production integrations before go-live.
- Keep compliance rules updated to your lending jurisdiction/regulatory requirements.
