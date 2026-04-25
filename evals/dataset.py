from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

SAMPLES = [
    {
        "query": "What should I do today?",
        "expected_tools": [ToolCall(name="query_notion_database", args={})]
    },
    {
        "query": "Open my notion",
        "expected_tools": [ToolCall(name="open_browser_url", args={"url": "https://www.notion.so"})]
    }
]