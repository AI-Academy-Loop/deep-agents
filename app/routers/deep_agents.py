from fastapi import APIRouter, HTTPException
from services.deep_agents_service import agent
import uuid
from langgraph.types import Command
from schemas import StartRun, ResumePayload

router = APIRouter()

def get_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}

@router.post("/run")
def start_run(payload: StartRun):
    thread_id = str(uuid.uuid4())
    config = get_config(thread_id)

    # Invoke the agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": payload.query}]},
        config=config # type: ignore
    )

    if result.get("__interrupt__"):
        # Extract interrupt information
        interrupts = result["__interrupt__"][0].value
        actions = interrupts["action_requests"]
        return {"status": "interrupted", "actions": actions, "thread_id": thread_id}
    else:
        # Completed without interrupt: return final answer
        if "messages" in result and result["messages"]:
            return {"status": "finished", "answer": result["messages"][-1]["content"]}
        else:
            raise HTTPException(status_code=500, detail="Agent failed to produce a final answer")

@router.post("/resume/{thread_id}")
def resume(thread_id: str, payload: ResumePayload):
    config = get_config(thread_id)

    # Convert payload to deepagents-compatible decisions
    decisions = []
    for d in payload.decisions:
        decision_obj = {"type": d.type}
        if d.type == "edit" and d.edited_action:
            decision_obj["edited_action"] = d.edited_action # type: ignore
        decisions.append(decision_obj)

    # Resume execution with decisions
    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config, # type: ignore
    )

    # Return final answer if available
    if "messages" in result and result["messages"]:
        return {"status": "finished", "answer": result}
    else:
        raise HTTPException(status_code=500, detail="Agent failed to produce a final answer")