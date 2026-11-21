import logging
import re
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """
    Production Knowledge Graph Store using Neo4j.
    Supports:
    1. Structured Entity-Relationship Triples: (:User)-[:RELATION]->(:Entity {name, type})
    2. Unstructured Fact Storage: (:User)-[:HAS_FACT]->(:MemoryFact {text, created_at})
    3. Multi-Hop Graph Traversal and Entity Querying.
    4. Belief Revision (Edge deletion/replacement on contradiction).
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
                logger.warning(f"Neo4j connectivity check failed: {e}. Graph operations will be gracefully skipped.")
                self.is_connected = False

    @staticmethod
    def _sanitize_relation(rel_type: str) -> str:
        """Sanitize relation name to valid UPPERCASE alphanumeric Cypher relationship type."""
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', rel_type.strip().upper())
        return clean if clean else "RELATED_TO"

    # =========================================================================
    # 1. Structured Triples (:User)-[:RELATION]->(:Entity)
    # =========================================================================
    def add_triple(self, user_id: str, relation: str, entity_name: str, entity_type: str = "Concept"):
        """
        Create a structured relationship in the graph:
        (:User {id}) -[:RELATION]-> (:Entity {name, type, updated_at})
        """
        if not self.is_connected or not self.driver:
            return

        rel = self._sanitize_relation(relation)
        query = f"""
        MERGE (u:User {{id: $user_id}})
        MERGE (e:Entity {{name: $entity_name}})
        ON CREATE SET e.type = $entity_type, e.created_at = timestamp()
        ON MATCH SET e.updated_at = timestamp()
        MERGE (u)-[r:{rel}]->(e)
        SET r.updated_at = timestamp()
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    user_id=user_id,
                    entity_name=entity_name.strip(),
                    entity_type=entity_type.strip()
                )
        except Exception as e:
            logger.warning(f"Failed to add triple in Neo4j: {e}")

    def remove_conflicting_triples(self, user_id: str, relation: str, new_entity_name: str):
        """
        Belief Revision: Removes prior single-valued relationships (e.g. if new LIVES_IN is 'Berlin',
        delete prior LIVES_IN edges to 'London').
        """
        if not self.is_connected or not self.driver:
            return

        rel = self._sanitize_relation(relation)
        query = f"""
        MATCH (u:User {{id: $user_id}})-[r:{rel}]->(e:Entity)
        WHERE toLower(e.name) <> toLower($new_entity_name)
        DELETE r
        """
        try:
            with self.driver.session() as session:
                session.run(query, user_id=user_id, new_entity_name=new_entity_name.strip())
        except Exception as e:
            logger.warning(f"Failed to remove conflicting triples in Neo4j: {e}")

    def get_user_triples(self, user_id: str, limit: int = 25) -> List[Dict[str, str]]:
        """
        Retrieve all structured triples (:User)-[r]->(:Entity) for a given user.
        """
        if not self.is_connected or not self.driver:
            return []

        query = """
        MATCH (u:User {id: $user_id})-[r]->(e:Entity)
        RETURN type(r) AS relation, e.name AS entity, e.type AS entity_type
        ORDER BY r.updated_at DESC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, user_id=user_id, limit=limit)
                return [
                    {
                        "relation": record["relation"],
                        "entity": record["entity"],
                        "type": record["entity_type"] or "Concept"
                    }
                    for record in result
                ]
        except Exception:
            return []

    def search_related_entities(self, user_id: str, keywords: List[str], limit: int = 10) -> List[Dict[str, str]]:
        """
        GraphRAG Retrieval: Find user entities whose names match query keywords.
        """
        if not self.is_connected or not self.driver or not keywords:
            return []

        query = """
        MATCH (u:User {id: $user_id})-[r]->(e:Entity)
        WHERE any(kw IN $keywords WHERE toLower(e.name) CONTAINS toLower(kw))
        RETURN type(r) AS relation, e.name AS entity, e.type AS entity_type
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, user_id=user_id, keywords=keywords, limit=limit)
                return [
                    {
                        "relation": record["relation"],
                        "entity": record["entity"],
                        "type": record["entity_type"] or "Concept"
                    }
                    for record in result
                ]
        except Exception:
            return []

    # =========================================================================
    # 2. Unstructured Fact Storage (:User)-[:HAS_FACT]->(:MemoryFact)
    # =========================================================================
    def add_fact(self, user_id: str, fact_text: str):
        """Create a (:User)-[:HAS_FACT]->(:MemoryFact) relationship."""
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
        """Retrieve all memory facts connected to a user in the graph."""
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

    # =========================================================================
    # 3. Cleanup & Teardown
    # =========================================================================
    def clear_user_graph(self, user_id: str):
        """Delete all facts, relationships, and connected entities for a user."""
        if not self.is_connected or not self.driver:
            return

        query = """
        MATCH (u:User {id: $user_id})-[r]->(n)
        DETACH DELETE n
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
