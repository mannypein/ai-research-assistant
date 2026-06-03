from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    
class AgentResponse(BaseModel):
    question: str
    answer: str
    sources_used: list[str]