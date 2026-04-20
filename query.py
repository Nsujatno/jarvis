import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# Load environment for OpenAI API Key
load_dotenv()

def query_db():
    # 1. Connect to the existing Chroma local folder
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("notion_docs")
    
    # 2. Setup the Vector Store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 3. Load the index from the existing storage
    # We don't need documents here because they are already in Chroma
    index = VectorStoreIndex.from_vector_store(
        vector_store
    )

    # 4. Initialize the Query Engine
    query_engine = index.as_query_engine()

    print("--- Notion Knowledge Base Ready ---")
    while True:
        user_query = input("\nAsk a question (or type 'exit'): ")
        if user_query.lower() in ['exit', 'quit']:
            break
            
        print("Searching...")
        response = query_engine.query(user_query)
        
        print("\n=== ANSWER ===")
        print(response)
        
        # Optional: Print the sources it used
        print("\n--- Sources ---")
        for node in response.source_nodes:
            title = node.metadata.get('name', 'Untitled')
            status = node.metadata.get('status', 'Unknown')
            print(f"- {title} (Status: {status})")

if __name__ == "__main__":
    query_db()
