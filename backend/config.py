import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "editorialcopilot")

# Enable LangSmith tracing if API key is present
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Models
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = "data"
BOOKS_DIR = os.path.join(DATA_DIR, "books")
VECTORSTORE_DIR = "vectorstore"

# Collections
BOOKS_COLLECTION = "books_fulltext"
CONTRACTS_COLLECTION = "contracts"
MEDIA_COLLECTION = "media"
EXTERNAL_BOOKS_COLLECTION = "external_books"
