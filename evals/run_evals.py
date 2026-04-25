import asyncio
import config
from ragas.metrics import ToolCallAccuracy, _AgentGoalAccuracyWithoutReference
from ragas.llms import llm_factory
from openai import OpenAI
from evals.dataset import SAMPLES

client = OpenAI()
llm = llm_factory("gpt-4o-mini", provider="openai", client=client)

async def run():
    tool_accuracy = ToolCallAccuracy()
    goal_accuracy = _AgentGoalAccuracyWithoutReference(llm=llm)

    print(f"\n{'='*80}")
    print(f"{'JARVIS EVALUATION RUNNER':^80}")
    print(f"{'='*80}\n")

    for i, sample in enumerate(SAMPLES, 1):
        query = sample.user_input[0].content
        
        # Extract predicted vs reference tool calls for visibility
        pred_tools = [tc.name for msg in sample.user_input if hasattr(msg, 'tool_calls') and msg.tool_calls for tc in msg.tool_calls]
        ref_tools = [tc.name for tc in (sample.reference_tool_calls or [])]
        
        # Get final response
        final_resp = sample.user_input[-1].content if sample.user_input else "N/A"

        print(f"--- Sample #{i} ---")
        print(f"Query:    {query}")
        print(f"Pred Tools: {pred_tools}")
        print(f"Ref Tools:  {ref_tools}")
        print(f"Response:   {final_resp[:100]}..." if len(final_resp) > 100 else f"Response:   {final_resp}")

        # Run scoring
        t_score = await tool_accuracy.multi_turn_ascore(sample)
        g_score = await goal_accuracy.multi_turn_ascore(sample)
        
        print(f"Result:     ToolAcc: {t_score:.2f} | GoalAcc: {g_score:.2f}")
        print(f"{'-'*40}\n")

if __name__ == "__main__":
    asyncio.run(run())
