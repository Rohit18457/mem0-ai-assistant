from typing import List, Dict, Any, Tuple, Optional
from groq import Groq
from mem0 import Memory
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    DEFAULT_USER_ID,
    get_mem0_config
)


class MemoryAssistant:
    """
    AI Assistant equipped with long-term memory powered by Mem0,
    Groq Cloud LLM, Hugging Face local embeddings, and Qdrant vector store.
    """

    def __init__(self, user_id: str = DEFAULT_USER_ID):
        self.user_id = user_id
        self.client = Groq(api_key=GROQ_API_KEY)
        self.mem_client = Memory.from_config(get_mem0_config())

    def search_memories(self, query: str, limit: int = 5) -> List[str]:
        """
        Search for relevant memories matching the user query.
        """
        try:
            search_response = self.mem_client.search(
                query=query,
                filters={"user_id": self.user_id},
                top_k=limit
            )
            results = (
                search_response.get("results", [])
                if isinstance(search_response, dict)
                else search_response
            )
            return [
                mem.get("memory")
                for mem in results
                if isinstance(mem, dict) and mem.get("memory")
            ]
        except Exception as e:
            # Fallback gracefully if search fails
            return []

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Retrieve all stored memories for the current user.
        """
        try:
            response = self.mem_client.get_all(filters={"user_id": self.user_id})
            if isinstance(response, dict):
                return response.get("results", [])
            return response if isinstance(response, list) else []
        except Exception:
            return []

    def chat(self, user_query: str) -> Tuple[str, List[str]]:
        """
        Process a user query by:
        1. Retrieving relevant memories from Qdrant.
        2. Augmenting the LLM system prompt with context.
        3. Generating an AI completion via Groq.
        4. Storing the conversation into memory.

        Returns:
            Tuple[str, List[str]]: (AI Response Text, List of retrieved memory snippets)
        """
        # 1. Retrieve relevant memories
        relevant_memories = self.search_memories(user_query)

        # 2. Build system prompt
        if relevant_memories:
            context_str = "\n".join(f"- {m}" for m in relevant_memories)
            system_prompt = (
                "You are an intelligent, personalized AI assistant with long-term memory.\n"
                f"Relevant context known about the user:\n{context_str}\n\n"
                "Use this context naturally to personalize your answers when relevant."
            )
        else:
            system_prompt = "You are an intelligent, personalized AI assistant with long-term memory."

        # 3. Request LLM response from Groq
        ai_response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        response_text = ai_response.choices[0].message.content

        # 4. Save interaction to Mem0
        # infer=False ensures fast, zero-token extraction using local Hugging Face embeddings
        try:
            self.mem_client.add(
                user_id=self.user_id,
                messages=[
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": response_text}
                ],
                infer=False
            )
        except Exception as err:
            # Log failure without disrupting user experience
            pass

        return response_text, relevant_memories

