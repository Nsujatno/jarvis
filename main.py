import asyncio
import config
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.workflow import Context
from utils.voice import VoiceListener

async def main():
    # setup agent via config
    agent = config.setup_agent()
    
    # Create a context to store the conversation history/session state
    ctx = Context(agent)

    print("\n=== Jarvis AI Agent (Notion) ===")
    
    # Voice Mode prompt
    voice_choice = input("Enable voice mode? (y/n): ").lower().strip()
    voice_enabled = voice_choice == 'y' or voice_choice == 'yes'
    listener = VoiceListener() if voice_enabled else None

    if voice_enabled:
        print("Voice mode active. Speak naturally, Jarvis will detect when you finish.")
    else:
        print("Text mode active. Type your requests below.")

    while True:
        if voice_enabled:
            # Get input from microphone
            user_input = await listener.get_input()
            if not user_input:
                continue
            print(f"\nYou (Voice): {user_input}")
        else:
            # Get input from keyboard
            user_input = input("\nYou: ")
            
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye sir, have a good day.")
            break
            
        print("Thinking...")
        
        # run agent
        handler = agent.run(user_input, ctx=ctx)

        # streaming
        async for ev in handler.stream_events():
        # if isinstance(ev, ToolCallResult):
        #     print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
            if isinstance(ev, AgentStream):
                print(f"{ev.delta}", end="", flush=True)
        response = await handler
        
        print(f"\nJarvis: {response}")

if __name__ == "__main__":
    asyncio.run(main())
