from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import QuestionRequest, AgentResponse
from app.agent import run_agent

app = FastAPI(
    title="AI Research Assistant",
    description="An AI-powered research assistant that uses a LangChain agent with web search capabilities, powered by Groq and monitored with LangSmith.",
    version="1.0.0"
)

# Allow all origins for now so the API is accessible publicly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "AI Research Assistant is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=AgentResponse)
async def ask_question(request: QuestionRequest):
    """
    Submit a question to the AI research assistant.
    The agent will decide whether to answer from its own knowledge
    or search the web for current information.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    result = run_agent(request.question)
    
    if "An error occurred" in result["answer"]:
        raise HTTPException(status_code=500, detail=result["answer"])
    
    return AgentResponse(
        question=result["question"],
        answer=result["answer"],
        sources_used=result["sources_used"]
    )