from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from config import settings

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai",api_key=settings.GOOGLE_API_KEY)

@tool
def delete_file(path: str) -> str:
    """Delete a file from the filesystem."""
    return f"Deleted {path}"

@tool
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    return f"Contents of {path}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Sent email to {to}"

# A checkpointer is required for the agent to be able to pause and resume.
checkpointer = MemorySaver()

agent = create_deep_agent(
    model=llm,
    tools=[delete_file, read_file, send_email],
    checkpointer=checkpointer,
    interrupt_on={
        "delete_file": True,  # Interrupt with default decisions: approve, edit, reject
        "read_file": False,   # No interrupt
        "send_email": {"allowed_decisions": ["approve", "reject"]},  # Interrupt without edit
    },
    subagents=[],
)
