from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import agent


app = FastAPI(
    title="AI Research Agent",
    description="Agentic AI backend powered by Groq",
    version="1.0.0"
)


class AgentRequest(BaseModel):
    goal: str


@app.get("/")
def root():
    return {
        "message": "AI Research Agent API is running 🚀"
    }


@app.post("/agent")
def run_agent(request: AgentRequest):

    if not request.goal.strip():
        raise HTTPException(
            status_code=400,
            detail="Goal cannot be empty"
        )

    try:
        result = agent.run(request.goal)

        return {
            "goal": request.goal,
            "result": result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
