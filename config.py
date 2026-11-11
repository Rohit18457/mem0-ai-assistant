import os
import warnings
import logging
from dotenv import load_dotenv

# Silence third-party noise, telemetry, and warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["MEM0_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"

# Load environment variables
load_dotenv()

# Groq Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. Please add it to your .env file or environment variables."
    )

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# Qdrant Settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mem0_hf")

# Embeddings Settings (Local Hugging Face)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "multi-qa-MiniLM-L6-cos-v1")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "384"))

# Neo4j Graph Store Settings
NEO4J_URI = os.getenv("NEO4J_URI") or os.getenv("NEO_CONNECTION_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO_PASSWORD")

# User Settings
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "rohit_suryawanshi")

def get_mem0_config() -> dict:
    """Return the configuration dictionary for Mem0 Memory instance."""
    config = {
        "version": "v1.1",
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": EMBEDDING_MODEL
            }
        },
        "llm": {
            "provider": "groq",
            "config": {
                "api_key": GROQ_API_KEY,
                "model": GROQ_MODEL
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": QDRANT_COLLECTION,
                "embedding_model_dims": EMBEDDING_DIMS,
                "host": QDRANT_HOST,
                "port": QDRANT_PORT
            }
        }
    }

    # Add Neo4j Graph Store configuration if credentials are provided
    if NEO4J_URI and NEO4J_PASSWORD:
        config["graph_store"] = {
            "provider": "neo4j",
            "config": {
                "url": NEO4J_URI,
                "username": NEO4J_USERNAME,
                "password": NEO4J_PASSWORD
            }
        }

    return config
