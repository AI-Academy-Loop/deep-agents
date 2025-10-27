from typing import List, Dict, Any
from pydantic import BaseModel

class StartRun(BaseModel):
    query: str

class Decision(BaseModel):
    type: str  # "approve" | "reject" | "edit"
    edited_action: Dict[str, Any] | None = None  # Required for "edit"

class ResumePayload(BaseModel):
    decisions: List[Decision]
