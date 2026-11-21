import json
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
    Production-grade AI Memory Assistant featuring a tri-layer hybrid architecture:
    1. Short-Term Working Memory: In-memory sliding window of recent conversation turns.
    2. Long-Term Semantic Vector Memory: Mem0 + local HuggingFace embeddings + Qdrant.
    3. Knowledge Graph Store: Neo4j Aura (:User)-[:RELATION]->(:Entity) structured GraphRAG.

    Key Production Capabilities:
    - Selective Fact & Triple Extraction (~70 tokens, zero prompt bloat).
    - Belief Revision & Conflict Resolution (supersedes obsolete/contradictory facts).
    - Dual-Channel Hybrid Retrieval (Vector semantic search + Graph entity traversal).
    - Full Memory Lifecycle CRUD & User Isolation.
    """

    SINGLE_VALUED_RELATIONS = {
        "LIVES_IN", "LOCATED_IN", "WORKS_AT", "EMPLOYED_BY", "JOB_TITLE",
        "FAVORITE_LANGUAGE", "PRIMARY_DATABASE"
    }

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
    # 1. Selective Fact & Entity-Relation Extraction
    # =========================================================================
    def extract_structured_facts(self, user_query: str) -> Dict[str, Any]:
        """
        Lightweight LLM call to extract enduring facts AND structured Graph Triples.
        Consumes ~70 tokens, producing machine-readable JSON without Mem0 prompt bloat.
        Returns:
            {
                "facts": ["User lives in Pune, India"],
                "triples": [{"relation": "LIVES_IN", "entity": "Pune", "type": "Location"}]
            }
        """
        prompt = (
            "You are a memory extraction component for an AI assistant.\n"
            "Analyze the user message and extract ONLY enduring personal facts, background, "
            "preferences, or identity attributes.\n"
            "Format your output as strict JSON with this exact schema:\n"
            "{\n"
            '  "facts": ["concise third-person statement", ...],\n'
            '  "triples": [{"relation": "RELATION_NAME", "entity": "EntityName", "type": "EntityType"}, ...]\n'
            "}\n\n"
            "Rules:\n"
            "- If the message contains NO personal facts (e.g. greetings, general questions), return: "
            '{"facts": [], "triples": []}\n'
            "- Relations must be UPPERCASE (e.g. LIVES_IN, WORKS_AT, SKILLED_IN, PREFERS, OWNS).\n"
            "- Do not include markdown codeblocks or explanation. Return JSON only.\n\n"
            f'User message: "{user_query}"'
        )

        try:
            res = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            raw = res.choices[0].message.content.strip()
            data = json.loads(raw)
            return {
                "facts": data.get("facts", []),
                "triples": data.get("triples", [])
            }
        except Exception:
            return {"facts": [], "triples": []}

    def extract_facts(self, user_query: str) -> List[str]:
        """Backward-compatible helper returning list of textual facts."""
        extracted = self.extract_structured_facts(user_query)
        return extracted.get("facts", [])

    # =========================================================================
    # 2. Belief Revision & Conflict Resolution (Interview Crown Jewel)
    # =========================================================================
    def resolve_conflicts(self, new_fact: str) -> List[str]:
        """
        Detects if a newly extracted fact contradicts or supersedes any prior memory.
        If a conflict exists, the outdated memory is removed from Qdrant to prevent
        contradictory prompt hallucinations.

        Example:
            Existing: "User lives in London"
            New: "User lives in Berlin"
            Action: Deletes "User lives in London" and logs belief revision.
        """
        conflicts_resolved = []
        try:
            # Search candidate memories in Qdrant relevant to this fact topic
            candidates = self.get_all_memories()
            if not candidates:
                return []

            candidate_texts = [
                {"id": m.get("id"), "text": m.get("memory", m.get("data", ""))}
                for m in candidates
                if m.get("id") and (m.get("memory") or m.get("data"))
            ]

            if not candidate_texts:
                return []

            # Quick verification LLM prompt to check for direct contradiction
            check_prompt = (
                "You are a belief revision engine. Determine if the NEW FACT directly contradicts "
                "or makes any of the EXISTING MEMORIES obsolete.\n\n"
                f"NEW FACT: \"{new_fact}\"\n\n"
                f"EXISTING MEMORIES:\n{json.dumps(candidate_texts, indent=2)}\n\n"
                "Return a JSON array of the IDs of any EXISTING memories that are contradicted or superseded by the NEW FACT.\n"
                "If none are contradicted, return an empty array: []\n"
                "Example format: [\"mem_id_1\"]"
            )

            res = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": check_prompt}],
                temperature=0.0,
                max_tokens=80,
                response_format={"type": "json_object"}
            )
            raw = res.choices[0].message.content.strip()
            parsed = json.loads(raw)
            conflicting_ids = parsed if isinstance(parsed, list) else parsed.get("contradicted_ids", parsed.get("ids", []))

            for cid in conflicting_ids:
                matching = next((m for m in candidate_texts if m["id"] == cid), None)
                if matching and self.delete_memory(cid):
                    note = f"Superseded prior memory: '{matching['text']}'"
                    conflicts_resolved.append(note)

        except Exception:
            pass

        return conflicts_resolved

    def is_duplicate(self, fact: str) -> bool:
        """Check if an identical or near-identical fact already exists in memory."""
        try:
            existing = self.search_memories(fact, limit=1)
            if not existing:
                return False
            top_mem = existing[0].lower().strip()
            cand = fact.lower().strip()
            return top_mem == cand or cand in top_mem or top_mem in cand
        except Exception:
            return False

    # =========================================================================
    # 3. Memory Search & CRUD Operations
    # =========================================================================
    def search_memories(self, query: str, limit: int = 5) -> List[str]:
        """Search relevant memories matching the query from Qdrant vector store."""
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

    def get_graph_triples(self) -> List[Dict[str, str]]:
        """Fetch all structured triples (:User)-[:RELATION]->(:Entity) in Neo4j."""
        return self.graph_store.get_user_triples(self.user_id)

    # =========================================================================
    # 4. Dual-Channel Hybrid Retrieval (Vector + GraphRAG Fusion)
    # =========================================================================
    def retrieve_hybrid_context(self, user_query: str) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Dual-channel retrieval:
        - Channel 1 (Vector): Semantically similar unstructured facts from Qdrant.
        - Channel 2 (Graph): Keyword-matched entity relations from Neo4j graph traversal.
        """
        # 1. Vector Search
        vector_memories = self.search_memories(user_query, limit=5)

        # 2. Graph Traversal (extract search keywords from user query)
        words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', user_query) if w.lower() not in {"what", "where", "when", "which", "your", "with", "have", "about"}]
        graph_triples = self.graph_store.search_related_entities(self.user_id, keywords=words, limit=8)

        return vector_memories, graph_triples

    def chat(self, user_query: str) -> Tuple[str, List[str], List[Dict[str, str]], List[str], List[str]]:
        """
        Production Hybrid Chat Turn:
        1. Retrieve dual-channel context (Qdrant semantic vector memories + Neo4j graph triples).
        2. Fuses context into system prompt.
        3. Appends recent sliding window of dialog turns.
        4. Synthesizes AI completion with Groq LPU.
        5. Selectively extracts new facts & triples.
        6. Runs Belief Revision / Conflict Resolution to supersede obsolete facts.
        7. Persists new facts to Qdrant and structured triples to Neo4j.

        Returns:
            Tuple[ai_response, vector_memories, graph_triples, saved_facts, conflicts_resolved]
        """
        # 1. Dual-Channel Retrieval
        vector_memories, graph_triples = self.retrieve_hybrid_context(user_query)

        # 2. Build fused system prompt
        context_sections = []
        if vector_memories:
            v_str = "\n".join(f"- {m}" for m in vector_memories)
            context_sections.append(f"Semantic Memories (Qdrant):\n{v_str}")

        if graph_triples:
            g_str = "\n".join(f"- User --[{t['relation']}]--> {t['entity']} ({t['type']})" for t in graph_triples)
            context_sections.append(f"Knowledge Graph Triples (Neo4j):\n{g_str}")

        if context_sections:
            fused_context = "\n\n".join(context_sections)
            system_prompt = (
                "You are an intelligent, personalized AI assistant with persistent long-term memory.\n"
                f"=== RECALLED USER KNOWLEDGE ===\n{fused_context}\n================================\n"
                "Personalize your answers accurately using the recalled knowledge."
            )
        else:
            system_prompt = "You are an intelligent, personalized AI assistant with persistent memory."

        # 3. Assemble messages with Short-Term Working Memory
        messages = [{"role": "system", "content": system_prompt}]
        window_size = self.max_history_turns * 2
        messages.extend(self.session_history[-window_size:])
        messages.append({"role": "user", "content": user_query})

        # 4. Generate AI response
        ai_response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages
        )
        response_text = ai_response.choices[0].message.content

        # 5. Extract enduring facts & structured triples
        extraction = self.extract_structured_facts(user_query)
        new_facts = extraction.get("facts", [])
        new_triples = extraction.get("triples", [])

        saved_facts = []
        conflicts_resolved = []

        # 6. Belief Revision & Persistence
        for fact in new_facts:
            # Check and resolve contradictions with existing memories
            resolved = self.resolve_conflicts(fact)
            conflicts_resolved.extend(resolved)

            if not self.is_duplicate(fact):
                try:
                    # Persist to Qdrant vector store
                    self.mem_client.add(fact, user_id=self.user_id, infer=False)
                    # Persist as unstructured fact node in Neo4j
                    self.graph_store.add_fact(self.user_id, fact)
                    saved_facts.append(fact)
                except Exception:
                    pass

        # 7. Persist structured Graph Triples in Neo4j
        for trip in new_triples:
            rel = trip.get("relation", "RELATED_TO")
            ent = trip.get("entity", "")
            ent_type = trip.get("type", "Concept")

            if ent:
                # If relation is single-valued (e.g. LIVES_IN), remove prior conflicting edges
                if rel in self.SINGLE_VALUED_RELATIONS:
                    self.graph_store.remove_conflicting_triples(self.user_id, rel, ent)

                self.graph_store.add_triple(self.user_id, rel, ent, ent_type)

        # 8. Update working session memory
        self.session_history.append({"role": "user", "content": user_query})
        self.session_history.append({"role": "assistant", "content": response_text})

        return response_text, vector_memories, graph_triples, saved_facts, conflicts_resolved
