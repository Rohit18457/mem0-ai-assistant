import logging
from typing import List, Optional
from neo4j import GraphDatabase, Driver
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """
    Dedicated Knowledge Graph Store using Neo4j.
    Connects to Neo4j Aura (or local Neo4j) to store and query
    structured entity relationships (:User)-[:HAS_FACT]->(:MemoryFact).
    """

    def __init__(self):
        self.driver: Optional[Driver] = None
        self.is_connected = False

        if NEO4J_URI and NEO4J_PASSWORD:
            try:
                self.driver = GraphDatabase.driver(
                    NEO4J_URI,
                    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
                )
                self.driver.verify_connectivity()
                self.is_connected = True
            except Exception as e:
                # Degrade gracefully without crashing the application
                logger.warning(f"Neo4j connectivity check failed: {e}. Graph operations will be skipped.")
                self.is_connected = False

    def add_fact(self, user_id: str, fact_text: str):
        """
        Create a (:User)-[:HAS_FACT]->(:MemoryFact) relationship in Neo4j.
        """
        if not self.is_connected or not self.driver:
            return

        query = """
        MERGE (u:User {id: $user_id})
        CREATE (m:MemoryFact {text: $fact_text, created_at: timestamp()})
        CREATE (u)-[:HAS_FACT]->(m)
        """
        try:
            with self.driver.session() as session:
                session.run(query, user_id=user_id, fact_text=fact_text)
        except Exception as e:
            logger.warning(f"Failed to record fact in Neo4j: {e}")

    def get_user_facts(self, user_id: str, limit: int = 15) -> List[str]:
        """
        Retrieve all memory facts connected to a user in the graph.
        """
        if not self.is_connected or not self.driver:
            return []

        query = """
        MATCH (u:User {id: $user_id})-[:HAS_FACT]->(m:MemoryFact)
        RETURN m.text AS fact
        ORDER BY m.created_at DESC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, user_id=user_id, limit=limit)
                return [record["fact"] for record in result]
        except Exception:
            return []

    def clear_user_graph(self, user_id: str):
        """
        Delete all facts and relationships for a user in the graph.
        """
        if not self.is_connected or not self.driver:
            return

        query = """
        MATCH (u:User {id: $user_id})-[r:HAS_FACT]->(m:MemoryFact)
        DETACH DELETE m
        """
        try:
            with self.driver.session() as session:
                session.run(query, user_id=user_id)
        except Exception:
            pass

    def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            try:
                self.driver.close()
            except Exception:
                pass
