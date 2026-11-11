import re
from typing import List, Dict, Any, Tuple, Optional
from groq import Groq
from mem0 import Memory
from graph_store import Neo4jGraphStore
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    DEFAULT_USER_ID,
    get_mem0_config
)


class MemoryAssistant:
    """
    Production-grade AI Assistant featuring a hybrid memory architecture:
    1. Short-Term Working Memory: In-memory sliding window of current session dialog turns.
    2. Long-Term Semantic Vector Memory: Mem0 + local Hugging Face embeddings + Qdrant.
    3. Knowledge Graph Store: Neo4j Aura (:User)-[:HAS_FACT]->(:MemoryFact).
    Includes selective fact extraction, deduplication, and full CRUD memory management.
    """

    def __init__(self, user_id: str = DEFAULT_USER_ID, max_history_turns: int = 5):
        self.user_id = self._validate_user_id(user_id)
        self.max_history_turns = max_history_turns
        self.session_history: List[Dict[str, str]] = []

        self.client = Groq(api_key=GROQ_API_KEY)
        self.mem_client = Memory.from_config(get_mem0_config())
        self.graph_store = Neo4jGraphStore()

    @staticmethod
    def _validate_user_id(user_id: str) -> str:
        """Validate user ID format (alphanumeric, underscores, hyphens, no spaces)."""
        clean_id = user_id.strip()
        if not clean_id or not re.match(r'^[a-zA-Z0-9_\-]+$', clean_id):
            raise ValueError(
                f"Invalid user_id '{user_id}'. Must be alphanumeric with no spaces (e.g. 'john_doe')."
            )
        return clean_id

    def set_user(self, new_user_id: str):
        """Switch active user profile and reset session working history."""
        self.user_id = self._validate_user_id(new_user_id)
        self.session_history.clear()

    def clear_session_history(self):
        """Reset short-term working memory without affecting persistent storage."""
        self.session_history.clear()

    # =========================================================================
    # 1. Selective Memory Extraction & Deduplication
    # =========================================================================
    def extract_facts(self, user_query: str) -> List[str]:
        """
        Lightweight LLM call to selectively extract only enduring facts,
        preferences, or personal attributes from user input.
        Consumes ~60 tokens, avoiding Mem0's massive 8k system prompt.
        """
        prompt = (
            "Analyze the following user message. Extract only enduring personal facts, "
            "preferences, skills, background, or identity details stated by the user.\n"
            "Rules:\n"
            "- Output concise, standalone third-person facts (e.g., 'User lives in London', 'User codes in Python').\n"
            "- If the message is only greetings, pleasantries, general questions, or contains NO personal details, reply exactly with 'NONE'.\n"
            "- Output one fact per line without bullet points or numbers.\n\n"
            f"User message: \"{user_query}\"\n"
            "Extracted facts:"
        )

        try:
            res = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120
            )
            raw = res.choices[0].message.content.strip()
            if not raw or raw.upper() == "NONE":
                return []

            facts = [
                line.strip()
                for line in raw.splitlines()
                if line.strip() and line.strip().upper() != "NONE"
            ]
            return facts
        except Exception:
            return []

    def is_duplicate(self, fact: str, similarity_threshold: float = 0.85) -> bool:
        """
        Check if an identical or near-identical fact already exists in memory.
        """
        try:
            existing = self.search_memories(fact, limit=1)
            if not existing:
                return False
            # If the closest memory is almost identical to the candidate fact
            top_mem = existing[0].lower().strip()
            cand = fact.lower().strip()
            if top_mem == cand or cand in top_mem or top_mem in cand:
                return True
            return False
        except Exception:
            return False

    # =========================================================================
    # 2. Memory Search & CRUD Operations
    # =========================================================================
    def search_memories(self, query: str, limit: int = 5) -> List[str]:
        """Search relevant memories matching the query for the active user."""
        try:
            response = self.mem_client.search(
                query=query,
                filters={"user_id": self.user_id},
                top_k=limit
            )
            results = response.get("results", []) if isinstance(response, dict) else response
            return [
                mem.get("memory")
                for mem in results
                if isinstance(mem, dict) and mem.get("memory")
            ]
        except Exception:
            return []

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Retrieve all stored memories for the active user."""
        try:
            response = self.mem_client.get_all(filters={"user_id": self.user_id})
            if isinstance(response, dict):
                return response.get("results", [])
            return response if isinstance(response, list) else []
        except Exception:
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by its ID."""
        try:
            self.mem_client.delete(memory_id=memory_id)
            return True
        except Exception:
            return False

    def update_memory(self, memory_id: str, new_text: str) -> bool:
        """Update the content of an existing memory."""
        try:
            self.mem_client.update(memory_id=memory_id, data=new_text)
            return True
        except Exception:
            return False

    def clear_all_memories(self) -> bool:
        """Delete all memories belonging to the active user across Qdrant and Neo4j."""
        try:
            self.mem_client.delete_all(user_id=self.user_id)
            self.graph_store.clear_user_graph(self.user_id)
            return True
        except Exception:
            return False

    def get_graph_memories(self) -> List[str]:
        """Fetch all facts stored in the Neo4j Knowledge Graph for this user."""
        return self.graph_store.get_user_facts(self.user_id)

    # =========================================================================
    # 3. Chat Pipeline (Dual-Memory & Knowledge Graph Synthesis)
    # =========================================================================
    def chat(self, user_query: str) -> Tuple[str, List[str], List[str]]:
        """
        Hybrid-memory chat turn:
        1. Query Long-Term Memory (Qdrant) for relevant past facts.
        2. Construct system prompt with long-term context.
        3. Assemble messages containing recent short-term dialogue turns + current query.
        4. Request completion from Groq LLM.
        5. Extract enduring facts and selectively persist to Qdrant AND Neo4j.
        6. Append turn to session working history.

        Returns:
            Tuple[str, List[str], List[str]]:
                (AI Response, Retrieved Long-term Memories, Newly Extracted Facts)
        """
        # 1. Retrieve long-term memory context
        retrieved_memories = self.search_memories(user_query)

        # 2. Build system prompt
        if retrieved_memories:
            context_str = "\n".join(f"- {m}" for m in retrieved_memories)
            system_prompt = (
                "You are an intelligent, personalized AI assistant with long-term memory.\n"
                f"Relevant facts known about the user:\n{context_str}\n\n"
                "Personalize your responses using this context when relevant."
            )
        else:
            system_prompt = "You are an intelligent, personalized AI assistant with long-term memory."

        # 3. Assemble messages with Short-Term Working Memory
        messages = [{"role": "system", "content": system_prompt}]

        # Include sliding window of recent session turns
        window_size = self.max_history_turns * 2
        recent_turns = self.session_history[-window_size:]
        messages.extend(recent_turns)

        # Add current user turn
        messages.append({"role": "user", "content": user_query})

        # 4. Generate AI response
        ai_response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages
        )
        response_text = ai_response.choices[0].message.content

        # 5. Selective fact extraction & hybrid persistence (Qdrant + Neo4j)
        extracted_facts = self.extract_facts(user_query)
        saved_facts = []

        for fact in extracted_facts:
            if not self.is_duplicate(fact):
                try:
                    # Persist to Qdrant vector store
                    self.mem_client.add(
                        fact,
                        user_id=self.user_id,
                        infer=False
                    )
                    # Persist to Neo4j knowledge graph
                    self.graph_store.add_fact(self.user_id, fact)

                    saved_facts.append(fact)
                except Exception:
                    pass

        # 6. Update session short-term working memory
        self.session_history.append({"role": "user", "content": user_query})
        self.session_history.append({"role": "assistant", "content": response_text})

        return response_text, retrieved_memories, saved_facts
