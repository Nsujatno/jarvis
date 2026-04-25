from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

SAMPLES = [
    # should call notion tool
    MultiTurnSample(
        user_input=[
            HumanMessage(content="what should i do today"),
            AIMessage(
                content="Let me check your tasks.",
                tool_calls=[ToolCall(name="query_notion_database", args={})]
            ),
            ToolMessage(content="Hw 6 due tomorrow, Huffman Codes 3 days overdue..."),
            AIMessage(content="You should work on Hw 6 due tomorrow, and catch up on Huffman Codes which is overdue."),
        ],
        reference_tool_calls=[ToolCall(name="query_notion_database", args={})]
    ),

    # should call browser tool
    MultiTurnSample(
        user_input=[
            HumanMessage(content="open my notion"),
            AIMessage(
                content="Opening Notion now.",
                tool_calls=[ToolCall(name="open_browser_url", args={"url": "https://notion.so"})]
            ),
            ToolMessage(content="Opening https://notion.so in your browser..."),
            AIMessage(content="Done, opening Notion in your browser."),
        ],
        reference_tool_calls=[ToolCall(name="open_browser_url", args={"url": "https://notion.so"})]
    ),

    # should call no tools
    MultiTurnSample(
        user_input=[
            HumanMessage(content="what is 2 + 2"),
            AIMessage(content="4"),
        ],
        reference_tool_calls=[]
    ),
]