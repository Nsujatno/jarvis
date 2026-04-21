import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import config
from tools.notion import query_database

def main():
    # chromadb setup
    db = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # notion polling via tool
    pages = query_database()
    for page in pages:
        print(page)

    # # store in chromadb
    # index = VectorStoreIndex.from_documents(
    #     documents, 
    #     storage_context=storage_context,
    #     show_progress=True
    # )


if __name__ == "__main__":
    main()