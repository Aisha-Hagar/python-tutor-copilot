import os
from pathlib import Path
from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")



# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = "Beginning_Programming_with_Python_For_Dummies.pdf"

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# PDF path
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / os.getenv("PDF_name")

# Vector dimension for nomic-embed-text (must match model)
VECTOR_DIM = 768