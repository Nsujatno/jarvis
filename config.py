import os
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import ReActAgent

load_dotenv()

from llama_index.core import PromptTemplate

# API Keys
NOTION_TOKEN = os.getenv("NOTION_INTEGRATION_TOKEN")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
_db_ids_str = os.getenv("NOTION_DATABASE_ID", "")
DATABASE_IDS = [i.strip() for i in _db_ids_str.split(",") if i.strip()]

# ChromaDB Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "notion_docs"

# Agent Configuration
LLM = OpenAI(model="gpt-4o-mini")

REACT_SYSTEM_HEADER_STR = """\
You are a helpful AI assistant named Jarvis.

## Tools
You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_desc}


## Output Format

Please answer in the same language as the question and use the following format:

Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})

Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

Please use a valid JSON format for the Action Input. Do NOT do this {{"input": "hello world", "num_beams": 5}}.

If this format is used, the tool will respond in the following format:

Observation: tool response

You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

Thought: I can answer without using any more tools. I'll use the user's language to answer.
Answer: [your answer here (In the same language as the user's question)]

Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.
"""

def setup_agent():
    """
    Initializes a ReActAgent with centralized LLM, prompts, and tools.
    USE LOCAL IMPORT to stop circular imports
    """
    from llama_index.core.tools import FunctionTool
    from tools.notion import query_database
    from tools.browser import open_url

    # Initialize tools
    notion_tool = FunctionTool.from_defaults(
        fn=query_database,
        name="query_notion_database",
        description="Queries the Notion database to get the latest list of tasks, courses, status, and content."
    )

    browser_tool = FunctionTool.from_defaults(
        fn=open_url,
        name="open_browser_url",
        description="Opens a given URL in the user's default web browser."
    )

    tools = [notion_tool, browser_tool]

    # Create agent
    agent = ReActAgent(tools=tools, llm=LLM, max_iterations=5)
    agent.update_prompts({"react_header": PromptTemplate(REACT_SYSTEM_HEADER_STR)})
    
    return agent
