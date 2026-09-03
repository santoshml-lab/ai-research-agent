from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import json

from agent import (
    agent,
    load_document,
    document_search
)


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="AI Research Agent",
    description="Agentic AI backend powered by Groq with RAG evaluation",
    version="1.1.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODEL
# =========================================================

class AgentRequest(BaseModel):

    goal: str

    # Optional ground-truth pages for RAG evaluation
    expected_pages: list[int] | None = Field(
        default=None,
        description="Relevant document page numbers used as ground truth for evaluation."
    )

class EvaluationRequest(BaseModel):

    questions: list[dict] = Field(
        ...,
        description=(
            "List of evaluation questions. "
            "Each item must contain question and expected_pages."
        )
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "AI Research Agent API is running 🚀"
    }


# =========================================================
# PDF UPLOAD / RAG
# =========================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # CHECK FILE TYPE
    # -----------------------------------------------------

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:

        # -------------------------------------------------
        # TEMPORARY FILE PATH
        # -------------------------------------------------

        file_path = f"/tmp/{file.filename}"

        # -------------------------------------------------
        # SAVE UPLOADED FILE
        # -------------------------------------------------

        contents = await file.read()

        with open(
            file_path,
            "wb"
        ) as buffer:

            buffer.write(contents)

        # -------------------------------------------------
        # LOAD DOCUMENT
        # -------------------------------------------------

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


# =========================================================
# RAG EVALUATION
# =========================================================

def evaluate_retrieval(
    query,
    expected_pages
):

    # -----------------------------------------------------
    # VALIDATE GROUND TRUTH
    # -----------------------------------------------------

    if not expected_pages:

        return {
            "status": "not_available",
            "message": (
                "Evaluation requires expected_pages "
                "as ground truth."
            )
        }

    try:

        expected_pages = set(
            int(page)
            for page in expected_pages
        )

        # -------------------------------------------------
        # RETRIEVE DOCUMENT CHUNKS
        # -------------------------------------------------

        search_result = document_search(
            query,
            top_k=3
        )

        # -------------------------------------------------
        # PARSE SEARCH RESULT
        # -------------------------------------------------

        if isinstance(
            search_result,
            str
        ):

            try:

                retrieved_chunks = json.loads(
                    search_result
                )

            except json.JSONDecodeError:

                return {
                    "status": "error",
                    "message": (
                        "Could not parse document "
                        "search results."
                    )
                }

        else:

            retrieved_chunks = search_result

        # -------------------------------------------------
        # CHECK SEARCH ERROR
        # -------------------------------------------------

        if not isinstance(
            retrieved_chunks,
            list
        ):

            return {
                "status": "error",
                "message": str(
                    search_result
                )
            }

        # -------------------------------------------------
        # RETRIEVED PAGES
        # -------------------------------------------------

        retrieved_pages = set()

        for item in retrieved_chunks:

            if isinstance(
                item,
                dict
            ):

                page = item.get(
                    "page"
                )

                if page is not None:

                    retrieved_pages.add(
                        int(page)
                    )

        # -------------------------------------------------
        # NO RETRIEVAL
        # -------------------------------------------------

        if not retrieved_pages:

            return {
                "status": "completed",

                "expected_pages": sorted(
                    expected_pages
                ),

                "retrieved_pages": [],

                "precision": 0.0,

                "recall": 0.0,

                "f1_score": 0.0
            }

        # -------------------------------------------------
        # TRUE POSITIVE PAGES
        # -------------------------------------------------

        true_positive = (
            expected_pages
            &
            retrieved_pages
        )

        # -------------------------------------------------
        # FALSE POSITIVE PAGES
        # -------------------------------------------------

        false_positive = (
            retrieved_pages
            -
            expected_pages
        )

        # -------------------------------------------------
        # FALSE NEGATIVE PAGES
        # -------------------------------------------------

        false_negative = (
            expected_pages
            -
            retrieved_pages
        )

        # -------------------------------------------------
        # PRECISION
        # -------------------------------------------------

        if retrieved_pages:

            precision = (
                len(true_positive)
                /
                len(retrieved_pages)
            )

        else:

            precision = 0.0

        # -------------------------------------------------
        # RECALL
        # -------------------------------------------------

        if expected_pages:

            recall = (
                len(true_positive)
                /
                len(expected_pages)
            )

        else:

            recall = 0.0

        # -------------------------------------------------
        # F1 SCORE
        # -------------------------------------------------

        if (
            precision + recall
            > 0
        ):

            f1_score = (
                2
                *
                precision
                *
                recall
                /
                (
                    precision
                    +
                    recall
                )
            )

        else:

            f1_score = 0.0

        # -------------------------------------------------
        # RETURN EVALUATION
        # -------------------------------------------------

        return {

            "status": "completed",

            "expected_pages": sorted(
                expected_pages
            ),

            "retrieved_pages": sorted(
                retrieved_pages
            ),

            "true_positive_pages": sorted(
                true_positive
            ),

            "false_positive_pages": sorted(
                false_positive
            ),

            "false_negative_pages": sorted(
                false_negative
            ),

            "precision": round(
                precision,
                4
            ),

            "recall": round(
                recall,
                4
            ),

            "f1_score": round(
                f1_score,
                4
            )
        }

    except Exception as error:

        return {

            "status": "error",

            "message": (
                f"Evaluation error: {error}"
            )
        }

def evaluate_multiple(
    questions
):

    results = []

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    completed_tests = 0

    for item in questions:

        question = item.get(
            "question"
        )

        expected_pages = item.get(
            "expected_pages"
        )

        if not question or not expected_pages:

            results.append(
                {
                    "status": "error",
                    "message": (
                        "Each test must contain "
                        "question and expected_pages."
                    )
                }
            )

            continue

        evaluation = evaluate_retrieval(
            question,
            expected_pages
        )

        results.append(
            {
                "question": question,
                **evaluation
            }
        )

        if evaluation.get(
            "status"
        ) == "completed":

            precision = evaluation.get(
                "precision",
                0.0
            )

            recall = evaluation.get(
                "recall",
                0.0
            )

            f1_score = evaluation.get(
                "f1_score",
                0.0
            )

            total_precision += precision
            total_recall += recall
            total_f1 += f1_score

            completed_tests += 1

    if completed_tests > 0:

        average_precision = (
            total_precision
            /
            completed_tests
        )

        average_recall = (
            total_recall
            /
            completed_tests
        )

        average_f1 = (
            total_f1
            /
            completed_tests
        )

    else:

        average_precision = 0.0
        average_recall = 0.0
        average_f1 = 0.0

    return {

        "status": "completed",

        "total_tests": len(
            questions
        ),

        "completed_tests": completed_tests,

        "average_precision": round(
            average_precision,
            4
        ),

        "average_recall": round(
            average_recall,
            4
        ),

        "average_f1_score": round(
            average_f1,
            4
        ),

        "tests": results
    }

# =========================================================
# MULTI-QUESTION RAG EVALUATION
# =========================================================

@app.post("/evaluate")
def evaluate_agent(
    request: EvaluationRequest
):

    if not request.questions:

        raise HTTPException(
            status_code=400,
            detail="At least one evaluation question is required."
        )

    try:

        evaluation = evaluate_multiple(
            request.questions
        )

        return evaluation

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================================
# AGENT
# =========================================================

@app.post("/agent")
def run_agent(
    request: AgentRequest
):

    # -----------------------------------------------------
    # VALIDATE GOAL
    # -----------------------------------------------------

    if not request.goal.strip():

        raise HTTPException(
            status_code=400,
            detail="Goal cannot be empty"
        )

    try:

        # -------------------------------------------------
        # RUN AGENT
        # -------------------------------------------------

        result = agent.run(
            request.goal
        )

        # -------------------------------------------------
        # BASE RESPONSE
        # -------------------------------------------------

        response = {

            "goal": request.goal,

            "result": result,

            "activity": agent.last_activity
        }

        # -------------------------------------------------
        # OPTIONAL EVALUATION
        # -------------------------------------------------

        if request.expected_pages is not None:

            response[
                "evaluation"
            ] = evaluate_retrieval(
                request.goal,
                request.expected_pages
            )

        return response

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
