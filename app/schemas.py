from typing import List, Dict, Optional
from pydantic import BaseModel

class UserMessage(BaseModel):
    role: str
    content: str

class StartExtraction(BaseModel):
    messages: List[UserMessage]
    mode: str = "cm"  # "cm" (diagnosis) | "pcs" (procedures)

class Decision(BaseModel):
    type: str  # "approve" | "reject" | "edit"
    edited: Optional[Dict] = None

class ResumePayload(BaseModel):
    decisions: List[Decision]