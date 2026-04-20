import os
import chromadb
from dotenv import load_dotenv
from llama_index.readers.notion import NotionPageReader
from notion_client import Client
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from utils.notion import extract_properties

load_dotenv()

database_id = ["25a9a54d4356807b817dd315db02e5d3"]


def main():
    # chromadb setup
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("notion_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # notion setup
    print("Authenticating and loading Notion data...")
    notion_token = os.getenv("NOTION_INTEGRATION_TOKEN")
    notion_reader = NotionPageReader(integration_token=notion_token)
    notion = Client(auth=notion_token)

    documents = notion_reader.load_data(database_ids=database_id)

    for document in documents:
        page_id = document.metadata.get('page_id')
        if page_id:
            # set document id to page id to prevent duplicates in our chroma db
            document.id_ = page_id
            try:
                page_data = notion.pages.retrieve(page_id=page_id)
                info = extract_properties(page_data)
                document.metadata.update(info)
                print(info)
            except Exception as e:
                print(f"Error fetching title for {page_id}: {e}")

    # store in chromadb
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )


if __name__ == "__main__":
    main()