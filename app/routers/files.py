from fastapi import APIRouter, File, UploadFile
import os
from typing import List
from app.utils.parsing import load_local_files, DATA_DIR, CHROMA_PERSIST_DIR
from app.schemas import ResumePayload, StartExtraction
import uuid
from app.services.deep_agents_service import agent, get_config
from langgraph.types import Command
import shutil


router = APIRouter()

@router.post("/load-local-files")
async def load_files():
    """Load ICD-10 files (will use cached vector stores if available)"""
    load_local_files()
    return {"status": "Local files loaded!"}

@router.post("/rebuild-vector-stores")
async def rebuild_stores():
    """Force rebuild of vector stores from source files (deletes cache)"""
    # Delete existing persistent stores
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print(f"🗑️  Deleted existing vector stores at {CHROMA_PERSIST_DIR}")
    
    # Rebuild from source
    load_local_files()
    return {"status": "Vector stores rebuilt from source files!"}
    

@router.post("/upload-files")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload YOUR 11 files."""
    for file in files:
        path = os.path.join(DATA_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
    load_local_files()
    return {"status": f"Loaded {len(files)} files!"}


@router.post("/extract")
def extract(payload: StartExtraction):
    thread_id = str(uuid.uuid4())
    config = get_config(thread_id)
    result = agent.invoke({"messages": [m.dict() for m in payload.messages]}, config=config)
    
    if result.get("__interrupt__"):
        interrupts = result["__interrupt__"][0].value
        action_requests = interrupts["action_requests"]
        
        # Enhance suggestions with preview data
        enhanced_suggestions = []
        for action in action_requests:
            tool_name = action.get("name")
            args = action.get("args", {})
            
            # Get preview of what the tool would return
            preview = ""
            if tool_name == "search_cm_codes":
                from app.services.deep_agents_service import search_cm_codes
                preview_result = search_cm_codes.invoke(args)
                import json
                try:
                    codes_data = json.loads(preview_result)
                    preview = "Found codes: " + ", ".join([f"{c.get('code', 'N/A')} - {c.get('description', 'N/A')[:50]}" for c in codes_data[:3]])
                except:
                    preview = preview_result[:200]
            elif tool_name == "extract_cm_guidelines":
                from app.services.deep_agents_service import extract_cm_guidelines
                preview_result = extract_cm_guidelines.invoke(args)
                preview = preview_result[:200] + "..." if len(preview_result) > 200 else preview_result
            elif tool_name == "search_pcs_codes":
                from app.services.deep_agents_service import search_pcs_codes
                preview_result = search_pcs_codes.invoke(args)
                import json
                try:
                    pcs_data = json.loads(preview_result)
                    preview = "Found procedures: " + ", ".join([f"{p.get('table_code', 'N/A')}" for p in pcs_data[:3]])
                except:
                    preview = preview_result[:200]
            
            enhanced_suggestions.append({
                "name": tool_name,
                "args": args,
                "description": f"Tool: {tool_name}\nQuery: {args.get('query', 'N/A')}\n\nPreview:\n{preview}"
            })
        
        return {"status": "interrupted", "suggestions": enhanced_suggestions, "thread_id": thread_id}
    else:
        last_msg = result["messages"][-1]
        return {"status": "finished", "output": last_msg.content}
    
@router.post("/resume/{thread_id}")
def resume(thread_id: str, payload: ResumePayload):
    config = get_config(thread_id)
    decisions = [{"type": d.type, **(d.edited or {})} for d in payload.decisions]
    result = agent.invoke(Command(resume={"decisions": decisions}), config=config)
    
    print(f"📊 Resume result keys: {result.keys()}")
    print(f"📨 Number of messages: {len(result.get('messages', []))}")
    
    # Check if there's another interruption
    if result.get("__interrupt__"):
        interrupts = result["__interrupt__"][0].value
        return {"status": "interrupted", "suggestions": interrupts["action_requests"], "thread_id": thread_id}
    else:
        messages = result.get("messages", [])
        if not messages:
            return {"status": "error", "output": "No messages in result"}
        
        last_msg = messages[-1]
        print(f"📤 Last message type: {type(last_msg)}")
        print(f"📝 Last message content: {last_msg.content if hasattr(last_msg, 'content') else last_msg}")
        
        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        return {"status": "finished", "output": content}