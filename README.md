<div align="center">

# 🧠 AI Hybrid Memory Assistant (v3.0 Enterprise)

### A Persistent, Context-Aware Agent with Dual-Channel Vector + Knowledge Graph (GraphRAG) Memory & Belief Revision
Built with **Mem0**, **Groq LPU LLM**, **Local Hugging Face Embeddings**, **Qdrant Vector Database**, **Neo4j Graph Database**, and **FastAPI**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Cloud-f55036.svg)](https://groq.com/)
[![Mem0](https://img.shields.io/badge/Memory-Mem0-purple.svg)](https://mem0.ai/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-dc2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Neo4j](https://img.shields.io/badge/Graph%20DB-Neo4j%20Aura-008cc1.svg?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-Hugging%20Face-ffd21e.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

</div>

## 📖 Overview

Standard LLMs are completely **stateless** — every conversation begins with a blank slate, forgetting user context, tech stacks, personal preferences, and instructions.

This project delivers an **Enterprise-Grade AI Hybrid Memory Architecture** combining:
1. **Short-Term Working Memory:** In-memory sliding window of active dialogue turns.
2. **Semantic Vector Memory (Qdrant):** Dense embeddings for conceptual and semantic similarity retrieval.
3. **Structured Knowledge Graph (Neo4j):** Explicit Subject-Predicate-Object triples `(:User)-[:RELATION]->(:Entity)` for multi-hop graph traversal (GraphRAG).
4. **Belief Revision Engine:** Automatically detects and resolves contradictions when user facts change (e.g., relocating cities or changing programming languages), eliminating hallucinated conflicts.
5. **Production REST API:** High-throughput FastAPI service with interactive Swagger documentation.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    User([👤 User Request]) --> Assistant[🧠 Hybrid Memory Assistant]

    subgraph Channel 1: Short-Term Working Memory
        Assistant <-->|Sliding Dialog Buffer| Buffer[(💭 Recent Dialog Window\nLast N Turns)]
    end

    subgraph Channel 2: Long-Term Semantic Vector Memory
        Assistant -->|1a. Embed Query| HF[🤗 Local HuggingFace Embedder\n384-d Cosine]
        HF -->|1b. Vector Search| Qdrant[(🗄️ Qdrant Vector DB\nlocalhost:6333)]
        Qdrant -->|1c. Semantic Memory Snippets| Fusion[🔀 Hybrid Context Fusion Engine]
    end

    subgraph Channel 3: Structured Knowledge Graph (GraphRAG)
        Assistant -->|2a. Entity Keyword Traversal| Neo4j[(🌐 Neo4j Graph Database\nAura Cloud)]
        Neo4j -->|2b. Structured Triples| Fusion
    end

    subgraph Cognitive Synthesis
        Fusion -->|Fused Context + Session Buffer| Groq[⚡ Groq LPU Engine\nqwen/qwen3.8-27b]
        Groq -->|Synthesized Response| Assistant
    end

    subgraph Fact Extraction & Belief Revision
        Assistant -->|User Turn| Extractor[🎯 Selective JSON Extractor\n~70 tokens]
        Extractor -->|Candidate Facts & Triples| ConflictCheck{Belief Conflict?}
        ConflictCheck -->|Contradiction Detected| Supersede[🔄 Invalidate Stale Memory in Qdrant & Neo4j]
        ConflictCheck -->|New Unique Fact| Persist[💾 Persist Vector to Qdrant & Triples to Neo4j]
        Supersede --> Persist
    end

    Assistant --> Output([💬 Personalized Response to User])
```

---

## ✨ Key Enterprise Capabilities

- **🔀 Dual-Channel Hybrid Retrieval (Vector + GraphRAG Fusion):**
  Combines unstructured semantic similarity from Qdrant with explicit multi-hop entity relationships from Neo4j (e.g. `User --[LIVES_IN]--> Pune`, `User --[SKILLED_IN]--> Python`).
- **🔄 Automated Belief Revision (Conflict Resolution):**
  Solves the classic vector DB pitfall where changing facts cause hallucinations. If the user previously stated *"I live in Seattle"* and later says *"I moved to Berlin"*, the belief revision engine supersedes the old memory in Qdrant and updates the Neo4j edge.
- **⚡ Zero Rate-Limit, Sub-Second Inference:**
  Combines Groq LPU's sub-second token generation with local HuggingFace embeddings (`multi-qa-MiniLM-L6-cos-v1`), incurring **zero embedding API costs**.
- **🌐 Production FastAPI REST API:**
  Enterprise endpoints (`/v1/chat`, `/v1/memories`, `/v1/graph`, `/v1/health`) with auto-generated Swagger UI at `/docs`.
- **📊 Scientific Benchmark Evaluation (`eval.py`):**
  Automated test suite measuring Recall@k, Response Grounding Rate, Latency (ms), and Belief Revision accuracy.

---

## 📂 Project Structure

```text
mem_agent/
├── config.py             # Centralized settings & Mem0 config generator
├── graph_store.py        # Neo4j Graph Database driver (triples & fact nodes)
├── memory_agent.py       # Core MemoryAssistant (hybrid retrieval & belief revision)
├── main.py               # Interactive CLI with Rich formatting & inspection commands
├── api.py                # Production FastAPI REST service with Swagger docs
├── eval.py               # Scientific evaluation suite for Recall, Latency & Conflict Resolution
├── docker-compose.yml    # Qdrant Vector DB container configuration
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── README.md             # Documentation & Interview Cheatsheet
```

---

## ⚡ Quickstart

### 1. Prerequisites
- **Python 3.10+**
- **Docker** (to run local Qdrant Vector DB)
- **Groq Cloud API Key** ([console.groq.com](https://console.groq.com/keys))
- **Neo4j Aura Database** ([neo4j.com/cloud/aura](https://neo4j.com/cloud/aura-free/))

### 2. Setup Environment

```bash
cd D:\Udemy_GenAI\mem_agent

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Qdrant Vector DB (Docker)

```bash
docker compose up -d
```
Verify Qdrant web dashboard at: `http://localhost:6333/dashboard`

### 4. Configure Credentials

Create `.env` from template:
```bash
cp .env.example .env
```
Ensure your `.env` contains:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

---

## 🚀 Usage

### Option A: Interactive CLI Assistant
```bash
python main.py
```
**Interactive Commands:**
- `/memories` — Inspect all long-term vector memories in Qdrant.
- `/triples` — Inspect structured Entity-Relationship Triples in Neo4j.
- `/graph` — Inspect memory fact nodes in Neo4j.
- `/user <id>` — Switch between isolated user profiles.
- `/reset` — Clear short-term working buffer while keeping long-term memory.
- `/clear` — Wipe all memories for active user across both DBs.

### Option B: Production REST API Server
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Open interactive Swagger UI: **`http://localhost:8000/docs`**

#### Sample API Request:
```bash
curl -X POST "http://localhost:8000/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "I prefer building microservices in Python and Go.", "user_id": "rohit"}'
```

### Option C: Run Scientific Evaluation Suite
```bash
python eval.py
```

---

## 🎯 Top 5 Memory Agent Interview Questions & System Design Answers

Prepare for technical interviews with this direct architecture breakdown:

### Q1: Why combine a Vector DB (Qdrant) with a Knowledge Graph (Neo4j) instead of using Vector search alone?
> **Answer:** Vector search calculates semantic proximity across dense embeddings, making it excellent for retrieving fuzzy, nuanced text passages. However, vector search struggles with:
> 1. Multi-hop relationship queries (e.g. *"What skills do people on my team have?"*).
> 2. Exact factual consistency (dense vectors can confuse negative or antonymous statements).
> 
> By implementing **Dual-Channel Hybrid Retrieval (GraphRAG)**, our agent queries Qdrant for semantic context and traverses Neo4j for deterministic triples `(:User)-[:SKILLED_IN]->(:Skill)`. We fuse both context streams before prompting the LLM, giving both fuzzy conceptual recall and exact relational grounding.

### Q2: What happens when user facts change over time? How do you prevent contradictory hallucinations?
> **Answer:** This is known as the **Belief Revision problem**. Naive vector memory agents blindly append new embeddings, so if a user previously said *"I live in Seattle"* and later says *"I moved to Berlin"*, both memories get retrieved, causing prompt hallucinations.
> 
> Our architecture implements an automated **Belief Revision Engine** (`resolve_conflicts` in `memory_agent.py`). When a new fact is extracted, the agent performs a conflict check against candidate memories. Contradicted memories are automatically deleted or marked as superseded in Qdrant, and the corresponding Neo4j edge is updated, guaranteeing single-truth consistency.

### Q3: How do you prevent token bloat and rate limits when extracting memories?
> **Answer:** Out-of-the-box frameworks like default Mem0 inject an ~8,000 token system prompt for memory inference on every chat turn, causing severe latency and token quota exhaustion.
> 
> We engineered a **Lightweight Selective Fact & Triple Extractor** (`extract_structured_facts`). It runs a dedicated, zero-shot structured JSON prompt consuming only ~70 tokens. Non-personal dialogue (e.g. *"Hello"*, *"Thank you"*) immediately returns empty lists, saving tokens and keeping retrieval latency under 150ms.

### Q4: How is user data isolated in production (Multi-Tenancy)?
> **Answer:** Multi-tenancy is enforced at every layer:
> 1. In Qdrant: Every payload stores `user_id`, and all search queries execute with strict payload filters: `{"must": [{"key": "user_id", "match": {"value": user_id}}]}`.
> 2. In Neo4j: All graph queries anchor to the root node `(:User {id: $user_id})` and traverse outwards along user-owned edges.
> 3. In the API layer: `user_id` is validated with strict regex alphanumeric bounds, preventing cross-tenant leakage.

### Q5: How do you evaluate and benchmark memory retrieval?
> **Answer:** We built a dedicated benchmark suite (`eval.py`) evaluating:
> 1. **Recall@k & Precision@k:** Measuring whether ground-truth memories are retrieved in the top $k$ candidates.
> 2. **Response Grounding Rate:** Verifying that the LLM's final generated answer explicitly cites the retrieved memory context.
> 3. **Retrieval Latency (ms):** Benchmarking vector search times (averaging ~10-25 ms with local embeddings).
> 4. **Belief Revision Success Rate:** Ingesting contradictory statements and asserting that outdated facts are superseded in subsequent turns.

---

## 📜 License

Distributed under the [MIT License](LICENSE).
