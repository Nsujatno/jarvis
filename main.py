import asyncio
import config
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import ReActAgent, AgentStream
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from tools.notion import query_database

async def main():
    # setup tools
    notion_tool = FunctionTool.from_defaults(
        fn=query_database,
        name="query_notion_database",
        description="Queries the Notion database to get the latest list of tasks, courses, status, and content."
    )

    # setup llm and agent
    llm = OpenAI(model="gpt-4o-mini")
    agent = ReActAgent(tools=[notion_tool], llm=llm)
    
    # Create a context to store the conversation history/session state
    ctx = Context(agent)

    print("\n=== Jarvis AI Agent (Notion) ===")
    print("Ask me about your tasks, courses, or deadlines.")
    print("Type 'exit' to quit.")

    while True:
        # Note: input() is synchronous, which is fine for a basic CLI loop
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye sir, have a good day.")
            break
            
        print("Thinking...")
        
        # 3. Run the Agent using the new Workflow pattern
        handler = agent.run(user_input, ctx=ctx)

        # streaming
        # async for ev in handler.stream_events():
        # # if isinstance(ev, ToolCallResult):
        # #     print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
        #     if isinstance(ev, AgentStream):
        #         print(f"{ev.delta}", end="", flush=True)
        response = await handler
        
        print(f"\nJarvis: {response}")

if __name__ == "__main__":
    asyncio.run(main())
