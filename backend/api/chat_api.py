from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Literal
from code.inference.chat_engine import generate_response
router = APIRouter()
model = None
tokenizer = None

def set_model_and_tokenizer(m, t):
    global model, tokenizer
    model = m
    tokenizer = t

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]

#POST
@router.post("/chat")
def chat(req: ChatRequest):

    messages = [m.model_dump() for m in req.messages]
    answer = generate_response(model, tokenizer, messages)
    return {"content": answer}