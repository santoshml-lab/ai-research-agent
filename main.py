from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import agent, load_document


app = FastAPI(
    title="AI Research Agent",
    description="Agentic AI backend powered by Groq",
    version="1.0.0"
)


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST MODEL
# =========================

class AgentRequest(BaseModel):
    goal: str


# =========================
# ROOT
# =========================

@app.get("/")
def root():

    return {
        "message": "AI Research Agent API is running 🚀"
    }


# =========================
# PDF UPLOAD / RAG
# =========================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    # Check file type

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:

        # Temporary file path

        file_path = f"/tmp/{file.filename}"

        # Save uploaded file

        contents = await file.read()

        with open(file_path, "wb") as buffer:

            buffer.write(contents)

        # Load document into RAG

        result = load_document(
            file_path
        )

        return {
            "filename": file.filename,
            "result": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================
# AGENT
# =========================

@app.post("/agent")
def run_agent(
    request: AgentRequest
):

    if not request.goal.strip():

        raise HTTPException(
            status_code=400,
            detail="Goal cannot be empty"
        )

    try:

        result = agent.run(
            request.goal
        )

        return {
            "goal": request.goal,
            "result": result,
            "activity": agent.last_activity
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
