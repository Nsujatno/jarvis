import config
from notion_client import Client
from llama_index.readers.notion import NotionPageReader
from utils.notion import extract_properties
from models.notion import NotionPage

def query_database():
    """
    queries notion database and returns a list of notionpage objects containing all relevant info
    Returns:
        list[NotionPage]: A list of validated NotionPage objects.
    """
    notion_token = config.NOTION_TOKEN
    notion_reader = NotionPageReader(integration_token=notion_token)
    notion = Client(auth=notion_token)

    print("Authenticating and loading Notion data...")
    documents = notion_reader.load_data(database_ids=config.DATABASE_IDS)
    
    results = []
    for document in documents:
        page_id = document.metadata.get('page_id')
        if page_id:
            try:
                page_data = notion.pages.retrieve(page_id=page_id)
                page_obj = extract_properties(page_data, document.text)
                if page_obj:
                    results.append(page_obj)
            except Exception as e:
                print(f"Error fetching detail for page {page_id}: {e}")
                
    return results