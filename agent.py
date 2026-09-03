import os
import json
import re

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# CLIENTS
# =========================================================

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# =========================================================
# CALCULATOR TOOL
# =========================================================

def calculator(expression):

    allowed = "0123456789+-*/(). "

    if not all(
        char in allowed
        for char in expression
    ):
        return "Invalid mathematical expression."

    try:

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:

        return "Could not calculate the expression."


# =========================================================
# WEB SEARCH TOOL
# =========================================================

def web_search(query):

    try:

        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5
        )

        results = response.get(
            "results",
            []
        )

        if not results:

            return "No search results found."

        formatted_results = []

        for result in results:

            formatted_results.append(
                {
                    "title": result.get(
                        "title",
                        ""
                    ),
                    "content": result.get(
                        "content",
                        ""
                    ),
                    "url": result.get(
                        "url",
                        ""
                    )
                }
            )

        return json.dumps(
            formatted_results,
            ensure_ascii=False
        )

    except Exception as error:

        return f"Search error: {error}"


# =========================================================
# RAG DOCUMENT STORE
# =========================================================

documents = []

word_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=1,
    sublinear_tf=True
)

document_vectors = None


# =========================================================
# CHUNK SETTINGS
# =========================================================

CHUNK_SIZE = 1200

CHUNK_OVERLAP = 200

DEFAULT_TOP_K = 3

MIN_RELEVANCE_SCORE = 0.05

# Only add a second chunk when its score is
# reasonably close to the best result.
SECONDARY_SCORE_RATIO = 0.65


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# CHUNK TEXT
# =========================================================

def create_chunks(
    text,
    page_number
):

    text = clean_text(text)

    if not text:

        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + CHUNK_SIZE,
            text_length
        )

        chunk_text = text[
            start:end
        ].strip()

        if chunk_text:

            chunks.append(
                {
                    "page": page_number,
                    "content": chunk_text,
                    "chunk_start": start,
                    "chunk_end": end
                }
            )

        if end >= text_length:

            break

        start = end - CHUNK_OVERLAP

    return chunks


# =========================================================
# LOAD DOCUMENT
# =========================================================

def load_document(file_path):

    global documents
    global document_vectors

    try:

        reader = PdfReader(
            file_path
        )

        new_documents = []

        # -------------------------------------------------
        # READ PDF PAGE BY PAGE
        # -------------------------------------------------

        for page_index, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text()

            if not page_text:

                continue

            page_chunks = create_chunks(
                page_text,
                page_index
            )

            new_documents.extend(
                page_chunks
            )

        # -------------------------------------------------
        # CHECK DOCUMENT
        # -------------------------------------------------

        if not new_documents:

            documents = []

            document_vectors = None

            return (
                "No readable text found "
                "in the document."
            )

        documents = new_documents

        # -------------------------------------------------
        # WORD TF-IDF
        # -------------------------------------------------

        word_vectors = (
            word_vectorizer.fit_transform(
                [
                    item["content"]
                    for item in documents
                ]
            )
        )

        # -------------------------------------------------
        # CHARACTER TF-IDF
        # -------------------------------------------------

        char_vectors = (
            char_vectorizer.fit_transform(
                [
                    item["content"]
                    for item in documents
                ]
            )
        )

        # -------------------------------------------------
        # COMBINE FEATURES
        # -------------------------------------------------

        document_vectors = hstack(
            [
                word_vectors,
                char_vectors
            ]
        ).tocsr()

        return (
            f"Document loaded successfully. "
            f"Pages: {len(reader.pages)}. "
            f"Created {len(documents)} "
            f"overlapping chunks."
        )

    except Exception as error:

        documents = []

        document_vectors = None

        return (
            f"Document loading error: {error}"
        )


# =========================================================
# DOCUMENT SEARCH TOOL
# =========================================================

def document_search(
    query,
    top_k=DEFAULT_TOP_K
):

    if (
        not documents
        or document_vectors is None
    ):

        return (
            "No document is currently loaded. "
            "Please load a document first."
        )

    try:

        # -------------------------------------------------
        # QUERY WORD VECTOR
        # -------------------------------------------------

        query_word_vector = (
            word_vectorizer.transform(
                [query]
            )
        )

        # -------------------------------------------------
        # QUERY CHARACTER VECTOR
        # -------------------------------------------------

        query_char_vector = (
            char_vectorizer.transform(
                [query]
            )
        )

        # -------------------------------------------------
        # COMBINE QUERY FEATURES
        # -------------------------------------------------

        query_vector = hstack(
            [
                query_word_vector,
                query_char_vector
            ]
        ).tocsr()

        # -------------------------------------------------
        # COSINE SIMILARITY
        # -------------------------------------------------

        similarities = cosine_similarity(
            query_vector,
            document_vectors
        )[0]

        # -------------------------------------------------
        # RANK ALL CHUNKS
        # -------------------------------------------------

        ranked_indexes = (
            similarities.argsort()[::-1]
        )

        # -------------------------------------------------
        # BEST SCORE
        # -------------------------------------------------

        best_score = float(
            similarities[
                ranked_indexes[0]
            ]
        )

        # -------------------------------------------------
        # SCORE THRESHOLD
        #
        # Dynamic threshold prevents weak chunks
        # from being included simply because top_k
        # has not been reached.
        # -------------------------------------------------

        dynamic_threshold = max(
            MIN_RELEVANCE_SCORE,
            best_score * SECONDARY_SCORE_RATIO
        )

        selected_indexes = []

        # -------------------------------------------------
        # FIRST PASS
        #
        # Select strongest chunks.
        # -------------------------------------------------

        for index in ranked_indexes:

            score = float(
                similarities[index]
            )

            if score < dynamic_threshold:

                continue

            selected_indexes.append(
                int(index)
            )

            if len(selected_indexes) >= top_k:

                break

        # -------------------------------------------------
        # FALLBACK
        # -------------------------------------------------

        if not selected_indexes:

            selected_indexes = [
                int(ranked_indexes[0])
            ]

        # -------------------------------------------------
        # PAGE DIVERSITY
        #
        # If multiple chunks come from the same page,
        # avoid filling the entire result set with
        # duplicate content from one page.
        # -------------------------------------------------

        final_indexes = []

        pages_seen = set()

        # First pass:
        # one strongest chunk per page.

        for index in selected_indexes:

            page = documents[index]["page"]

            if page in pages_seen:

                continue

            final_indexes.append(
                index
            )

            pages_seen.add(
                page
            )

        # Second pass:
        # allow additional chunk only if required.

        if len(final_indexes) < top_k:

            for index in selected_indexes:

                if index in final_indexes:

                    continue

                final_indexes.append(
                    index
                )

                if len(final_indexes) >= top_k:

                    break

        # -------------------------------------------------
        # SORT BY RELEVANCE
        # -------------------------------------------------

        final_indexes.sort(
            key=lambda index: similarities[index],
            reverse=True
        )

        # -------------------------------------------------
        # BUILD RESULTS
        # -------------------------------------------------

        results = []

        for index in final_indexes:

            results.append(
                {
                    "page": documents[index][
                        "page"
                    ],

                    "score": round(
                        float(
                            similarities[index]
                        ),
                        4
                    ),

                    "content": documents[index][
                        "content"
                    ][:900]
                }
            )

        # -------------------------------------------------
        # RETURN RESULTS
        # -------------------------------------------------

        return json.dumps(
            results,
            ensure_ascii=False
        )

    except Exception as error:

        return (
            f"Document search error: {error}"
        )


# =========================================================
# TOOLS
# =========================================================

tools = [

    # =====================================================
    # CALCULATOR
    # =====================================================

    {
        "type": "function",

        "function": {

            "name": "calculator",

            "description":
                "Calculate a mathematical expression.",

            "parameters": {

                "type": "object",

                "properties": {

                    "expression": {

                        "type": "string",

                        "description":
                            "Mathematical expression."
                    }
                },

                "required": [
                    "expression"
                ]
            }
        }
    },

    # =====================================================
    # WEB SEARCH
    # =====================================================

    {
        "type": "function",

        "function": {

            "name": "web_search",

            "description": (
                "Search the web for current "
                "or external information."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {

                        "type": "string",

                        "description":
                            "Search query."
                    }
                },

                "required": [
                    "query"
                ]
            }
        }
    },

    # =====================================================
    # DOCUMENT SEARCH
    # =====================================================

    {
        "type": "function",

        "function": {

            "name": "document_search",

            "description": (
                "Search the loaded PDF document "
                "for relevant information. "
                "Use this tool whenever the user's "
                "question depends on the uploaded "
                "document."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {

                        "type": "string",

                        "description":
                            "Question or search query "
                            "for the loaded document."
                    }
                },

                "required": [
                    "query"
                ]
            }
        }
    }
]


# =========================================================
# AI AGENT
# =========================================================

class ResearchAgent:

    def __init__(self):

        self.name = (
            "AI Research Agent"
        )

        self.conversation_history = []

        self.last_activity = []


    # =====================================================
    # RUN AGENT
    # =====================================================

    def run(self, goal):

        self.last_activity = []

        # -------------------------------------------------
        # SYSTEM MESSAGE
        # -------------------------------------------------

        messages = [

            {
                "role": "system",

                "content": (

                    "You are an AI Research Agent. "

                    "Understand the user's goal "
                    "and complete it using the "
                    "available tools when necessary. "

                    "Use calculator for mathematical "
                    "calculations. "

                    "Use web_search for current, "
                    "external, or web-based information. "

                    "Use document_search whenever "
                    "the answer should come from the "
                    "uploaded document. "

                    "You may use multiple tools "
                    "and multiple tool calls when "
                    "needed. "

                    "Continue working until the "
                    "user's goal is fully completed. "

                    "When answering questions about "
                    "the uploaded document, rely only "
                    "on information returned by "
                    "document_search. "

                    "Do not invent document facts. "

                    "If the retrieved document content "
                    "does not contain enough information "
                    "to answer confidently, clearly "
                    "say that the information could "
                    "not be found in the document. "

                    "When document_search returns page "
                    "numbers, mention relevant page "
                    "numbers in the answer. "

                    "When multiple retrieved sections "
                    "are relevant, combine them carefully. "

                    "Do not claim that a fact came from "
                    "the document unless it was retrieved "
                    "from the document. "

                    "When web_search is used, base the "
                    "answer on the returned results. "

                    "Include a Sources section when "
                    "web_search is used. "

                    "Use URLs exactly as returned by "
                    "the search tool. "

                    "Never invent sources or URLs. "

                    "Use previous conversation context "
                    "when it is relevant."
                )
            }
        ]

        # -------------------------------------------------
        # ADD MEMORY
        # -------------------------------------------------

        messages.extend(
            self.conversation_history
        )

        # -------------------------------------------------
        # CURRENT REQUEST
        # -------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": goal
            }
        )

        # =================================================
        # MULTI-STEP AGENT LOOP
        # =================================================

        max_iterations = 5

        for iteration in range(
            max_iterations
        ):

            try:

                response = (
                    groq_client
                    .chat
                    .completions
                    .create(

                        model="openai/gpt-oss-20b",

                        messages=messages,

                        tools=tools,

                        tool_choice="auto",

                        parallel_tool_calls=False,

                        temperature=0.2
                    )
                )

            except Exception as error:

                return (
                    f"Agent error: {error}"
                )

            message = (
                response.choices[0].message
            )

            # =================================================
            # FINAL ANSWER
            # =================================================

            if not message.tool_calls:

                final_answer = (
                    message.content
                    or "No response received."
                )

                # -------------------------------------------------
                # SAVE MEMORY
                # -------------------------------------------------

                self.conversation_history.append(
                    {
                        "role": "user",
                        "content": goal
                    }
                )

                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": final_answer
                    }
                )

                # -------------------------------------------------
                # KEEP LAST 4 MESSAGES
                # -------------------------------------------------

                self.conversation_history = (
                    self.conversation_history[-4:]
                )

                return final_answer

            # =================================================
            # ADD TOOL CALL MESSAGE
            # =================================================

            messages.append(
                message
            )

            # =================================================
            # EXECUTE TOOLS
            # =================================================

            for tool_call in message.tool_calls:

                function_name = (
                    tool_call.function.name
                )

                # =================================================
                # ACTIVITY MAP
                # =================================================

                activity_map = {

                    "calculator": {

                        "tool": "calculator",

                        "label": "Calculator",

                        "icon": "calculator"
                    },

                    "web_search": {

                        "tool": "web_search",

                        "label": "Web Search",

                        "icon": "globe"
                    },

                    "document_search": {

                        "tool": "document_search",

                        "label": "Document Search",

                        "icon": "file"
                    }
                }

                activity = activity_map.get(
                    function_name
                )

                if activity:

                    self.last_activity.append(
                        {
                            "step": len(
                                self.last_activity
                            ) + 1,

                            "tool": activity[
                                "tool"
                            ],

                            "label": activity[
                                "label"
                            ],

                            "icon": activity[
                                "icon"
                            ],

                            "status": "completed"
                        }
                    )

                # =================================================
                # PARSE TOOL ARGUMENTS
                # =================================================

                try:

                    arguments = json.loads(
                        tool_call.function.arguments
                    )

                except (
                    json.JSONDecodeError
                ):

                    result = (
                        "Invalid tool arguments."
                    )

                    messages.append(
                        {
                            "role": "tool",

                            "tool_call_id":
                                tool_call.id,

                            "content":
                                result
                        }
                    )

                    continue

                # =================================================
                # CALCULATOR
                # =================================================

                if function_name == "calculator":

                    result = calculator(
                        arguments.get(
                            "expression",
                            ""
                        )
                    )

                # =================================================
                # WEB SEARCH
                # =================================================

                elif function_name == "web_search":

                    result = web_search(
                        arguments.get(
                            "query",
                            ""
                        )
                    )

                # =================================================
                # DOCUMENT SEARCH
                # =================================================

                elif function_name == "document_search":

                    result = document_search(
                        arguments.get(
                            "query",
                            ""
                        )
                    )

                # =================================================
                # UNKNOWN TOOL
                # =================================================

                else:

                    result = (
                        "Unknown tool."
                    )

                # =================================================
                # RETURN TOOL RESULT
                # =================================================

                messages.append(
                    {
                        "role": "tool",

                        "tool_call_id":
                            tool_call.id,

                        "content":
                            result
                    }
                )

        # =================================================
        # MAX ITERATIONS
        # =================================================

        return (
            "The agent reached the maximum "
            "number of tool steps without "
            "completing the task."
        )


# =========================================================
# AGENT INSTANCE
# =========================================================

agent = ResearchAgent()


















    

        
