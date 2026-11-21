"""
Production FastAPI REST Service for AI Hybrid Memory Assistant.
Provides endpoints for:
- Chat completions with dual-channel memory retrieval and belief revision.
- Memory CRUD operations across Qdrant Vector DB.
- Knowledge Graph inspection from Neo4j.
- Healthcheck and service readiness probe.
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from memory_agent import MemoryAssistant
from config import DEFAULT_USER_ID, GROQ_MODEL, QDRANT_COLLECTION

app = FastAPI(
    title="🧠 AI Hybrid Memory Assistant API",
    description="Enterprise REST API featuring Short-Term Working Memory, Qdrant Vector Storage, and Neo4j GraphRAG.",
    version="3.0.0"
)

# CORS Middleware for web/frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache assistant instances by user_id
_assistants: Dict[str, MemoryAssistant] = {}


def get_assistant(user_id: str) -> MemoryAssistant:
    """Retrieve or initialize a MemoryAssistant for a given user."""
    clean_id = user_id.strip() if user_id else DEFAULT_USER_ID
    if clean_id not in _assistants:
        _assistants[clean_id] = MemoryAssistant(user_id=clean_id)
    return _assistants[clean_id]


# =============================================================================
# Pydantic Request & Response Schemas
# =============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., example="I prefer writing microservices in Go and Python.")
    user_id: Optional[str] = Field(default=DEFAULT_USER_ID, example="rohit_suryawanshi")


class GraphTriple(BaseModel):
    relation: str
    entity: str
    type: str


class ChatResponse(BaseModel):
    response: str
    user_id: str
    vector_memories_recalled: List[str]
    graph_triples_recalled: List[GraphTriple]
    new_facts_learned: List[str]
    conflicts_resolved: List[str]


class MemoryItem(BaseModel):
    id: str
    memory: str
    created_at: Optional[str] = None


class AddMemoryRequest(BaseModel):
    fact: str = Field(..., example="User loves drinking green tea.")
    user_id: Optional[str] = Field(default=DEFAULT_USER_ID)


class HealthResponse(BaseModel):
    status: str
    groq_model: str
    qdrant_collection: str
    neo4j_connected: bool


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/v1/health", response_model=HealthResponse, tags=["Health"])
def healthcheck():
    """Verify backend connectivity and dependency states."""
    assistant = get_assistant(DEFAULT_USER_ID)
    return HealthResponse(
        status="healthy",
        groq_model=GROQ_MODEL,
        qdrant_collection=QDRANT_COLLECTION,
        neo4j_connected=assistant.graph_store.is_connected
    )


@app.post("/v1/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Perform a conversational turn with the hybrid memory assistant:
    - Queries Qdrant semantic vectors + Neo4j graph triples.
    - Executes automated Belief Revision to supersede contradictory facts.
    - Synthesizes personalized completion using Groq LPU.
    """
    try:
        assistant = get_assistant(request.user_id)
        reply, v_mems, g_trips, saved, conflicts = assistant.chat(request.message)

        formatted_triples = [
            GraphTriple(relation=t["relation"], entity=t["entity"], type=t["type"])
            for t in g_trips
        ]

        return ChatResponse(
            response=reply,
            user_id=assistant.user_id,
            vector_memories_recalled=v_mems,
            graph_triples_recalled=formatted_triples,
            new_facts_learned=saved,
            conflicts_resolved=conflicts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/memories", response_model=List[MemoryItem], tags=["Memory Management"])
def list_memories(user_id: str = Query(default=DEFAULT_USER_ID)):
    """Retrieve all stored vector memories from Qdrant for a user."""
    try:
        assistant = get_assistant(user_id)
        raw_mems = assistant.get_all_memories()
        return [
            MemoryItem(
                id=m.get("id", ""),
                memory=m.get("memory", m.get("data", "")),
                created_at=str(m.get("created_at", ""))
            )
            for m in raw_mems
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories", status_code=status.HTTP_201_CREATED, tags=["Memory Management"])
def add_memory(request: AddMemoryRequest):
    """Manually inject a verified fact into long-term memory."""
    try:
        assistant = get_assistant(request.user_id)
        assistant.mem_client.add(request.fact, user_id=assistant.user_id, infer=False)
        assistant.graph_store.add_fact(assistant.user_id, request.fact)
        return {"message": "Memory persisted successfully.", "fact": request.fact}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/memories/{memory_id}", tags=["Memory Management"])
def delete_memory(memory_id: str, user_id: str = Query(default=DEFAULT_USER_ID)):
    """Delete a specific memory by its ID."""
    assistant = get_assistant(user_id)
    success = assistant.delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found or deletion failed.")
    return {"message": f"Memory '{memory_id}' deleted successfully."}


@app.delete("/v1/memories", tags=["Memory Management"])
def clear_all_memories(user_id: str = Query(default=DEFAULT_USER_ID)):
    """Delete all memories for a user across Qdrant and Neo4j."""
    assistant = get_assistant(user_id)
    assistant.clear_all_memories()
    return {"message": f"All memories for user '{user_id}' cleared."}


@app.get("/v1/graph", response_model=List[GraphTriple], tags=["Knowledge Graph"])
def get_graph(user_id: str = Query(default=DEFAULT_USER_ID)):
    """Inspect structured Entity-Relationship Triples from Neo4j."""
    assistant = get_assistant(user_id)
    triples = assistant.get_graph_triples()
    return [
        GraphTriple(relation=t["relation"], entity=t["entity"], type=t["type"])
        for t in triples
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
