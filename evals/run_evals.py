import asyncio
import config
import uuid
from ragas.metrics import ToolCallAccuracy, _AgentGoalAccuracyWithoutReference
from ragas.llms import llm_factory
from ragas.dataset_schema import MultiTurnSample
from openai import OpenAI
from evals.dataset import SAMPLES
from config import setup_agent
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentStream
from ragas.messages import HumanMessage, ToolCall, AIMessage, ToolMessage
client = OpenAI()
llm = llm_factory("gpt-4o-mini", provider="openai", client=client)

agent = setup_agent()

async def run():
    tool_accuracy = ToolCallAccuracy()
    goal_accuracy = _AgentGoalAccuracyWithoutReference(llm=llm)

    print(f"\n{'='*80}")
    print(f"{'JARVIS EVALUATION RUNNER':^80}")
    print(f"{'='*80}\n")

    # loop through each sample
    for i, sample in enumerate(SAMPLES, 1):
        # get query and expected tools
        query = sample["query"]
        expected_tools = sample["expected_tools"]
        
        # create new context on every run
        ctx = Context(agent)
        handler = agent.run(query, ctx=ctx)

        # for evals need: [HumanMessage, AIMessage(with tool calls), ToolMessage, AIMessage(final answer)]
        user_input = [HumanMessage(content=query)]

        async for ev in handler.stream_events():
            # streaming for me to debug
            if isinstance(ev, AgentStream):
                print(f"{ev.delta}", end="", flush=True)
            # if tool call, checking by class name avoids circular imports
            if type(ev).__name__ == "ToolCallResult":
                # generate random tool call id
                dummy_id = f"call_{uuid.uuid4().hex[:8]}"

                # record calling a tool
                user_input.append(AIMessage(
                    content="", 
                    tool_calls=[ToolCall(name=ev.tool_name, args=ev.tool_kwargs)]
                ))
                
                # record tool response
                user_input.append(ToolMessage(content=str(ev.tool_output), tool_call_id=dummy_id))
        
        # get final answer
        response = await handler
        user_input.append(AIMessage(content=str(response)))

        # dynamic sampling of 
        dynamic_sample = MultiTurnSample(
            user_input=user_input,
            reference_tool_calls=expected_tools
        )

        # Extract predicted vs reference tool calls for visibility
        pred_tools = [tc.name for msg in dynamic_sample.user_input if hasattr(msg, 'tool_calls') and msg.tool_calls for tc in msg.tool_calls]
        ref_tools = [tc.name for tc in (dynamic_sample.reference_tool_calls or [])]
        
        # Get final response
        final_resp = dynamic_sample.user_input[-1].content if dynamic_sample.user_input else "N/A"

        print(f"--- Sample #{i} ---")
        print(f"Query:    {query}")
        print(f"Pred Tools: {pred_tools}")
        print(f"Ref Tools:  {ref_tools}")
        print(f"Response:   {final_resp[:100]}..." if len(final_resp) > 100 else f"Response:   {final_resp}")

        # Run scoring using dynamic sampling
        t_score = await tool_accuracy.multi_turn_ascore(dynamic_sample)
        g_score = await goal_accuracy.multi_turn_ascore(dynamic_sample)
        
        print(f"Result:     ToolAcc: {t_score:.2f} | GoalAcc: {g_score:.2f}")
        print(f"{'-'*40}\n")



if __name__ == "__main__":
    asyncio.run(run())
