import os
from dotenv import load_dotenv

load_dotenv()

# Notion Configuration
NOTION_TOKEN = os.getenv("NOTION_INTEGRATION_TOKEN")
DATABASE_IDS = ["25a9a54d4356807b817dd315db02e5d3"]

# ChromaDB Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "notion_docs"
