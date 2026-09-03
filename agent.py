import os
import json
import re
import math

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

                    # Limit web content
                    # to avoid huge model requests
                    "content": result.get(
                        "content",
                        ""
                    )[:1800],

                    "url": result.get(
                        "url",
                        ""
                    )
                }
            )

        output = json.dumps(
            formatted_results,
            ensure_ascii=False
        )

        # Final safety limit
        return output[:7000]

    except Exception as error:

        return f"Search error: {error}"


# =========================================================
# RAG DOCUMENT STORE
# =========================================================

documents = []


# =========================================================
# WORD TF-IDF
# =========================================================

word_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)


# =========================================================
# CHARACTER TF-IDF
# =========================================================

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

MIN_RELEVANCE_SCORE = 0.03


# =========================================================
# MODEL CONTEXT LIMITS
# =========================================================

# Maximum number of characters sent from
# retrieved document context to the LLM.
MAX_DOCUMENT_CONTEXT_CHARS = 6500

# Maximum characters for one conversation message.
MAX_HISTORY_MESSAGE_CHARS = 1200

# Number of previous messages retained.
MAX_HISTORY_MESSAGES = 4

# Maximum number of tool iterations.
MAX_AGENT_ITERATIONS = 4


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
        # READ PAGE BY PAGE
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
        # DOCUMENT CONTENT
        # -------------------------------------------------

        contents = [
            item["content"]
            for item in documents
        ]

        # -------------------------------------------------
        # WORD TF-IDF
        # -------------------------------------------------

        word_vectors = (
            word_vectorizer.fit_transform(
                contents
            )
        )

        # -------------------------------------------------
        # CHARACTER TF-IDF
        # -------------------------------------------------

        char_vectors = (
            char_vectorizer.fit_transform(
                contents
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

        selected_indexes = []

        # -------------------------------------------------
        # SELECT TOP RELEVANT CHUNKS
        # -------------------------------------------------

        for index in ranked_indexes:

            score = float(
                similarities[index]
            )

            if score < MIN_RELEVANCE_SCORE:

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

            best_index = int(
                similarities.argmax()
            )

            selected_indexes = [
                best_index
            ]

        # -------------------------------------------------
        # ADD ADJACENT CHUNKS
        # -------------------------------------------------

        expanded_indexes = set(
            selected_indexes
        )

        for index in selected_indexes:

            previous_index = index - 1

            next_index = index + 1

            if (
                previous_index >= 0
                and previous_index < len(documents)
            ):

                expanded_indexes.add(
                    previous_index
                )

            if (
                next_index >= 0
                and next_index < len(documents)
            ):

                expanded_indexes.add(
                    next_index
                )

        # -------------------------------------------------
        # SORT DOCUMENT ORDER
        # -------------------------------------------------

        final_indexes = sorted(
            expanded_indexes
        )

        results = []

        current_chars = 0

        # -------------------------------------------------
        # BUILD BOUNDED CONTEXT
        # -------------------------------------------------

        for index in final_indexes:

            content = documents[index][
                "content"
            ]

            remaining = (
                MAX_DOCUMENT_CONTEXT_CHARS
                - current_chars
            )

            if remaining <= 0:

                break

            content = content[
                :min(
                    900,
                    remaining
                )
            ]

            if not content:

                continue

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

                    "content": content
                }
            )

            current_chars += len(
                content
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
# RETRIEVAL EVALUATION
# =========================================================
#
# Each evaluation item should look like:
#
# {
#     "query": "What is acceleration?",
#     "relevant_pages": [7]
# }
#
# OR:
#
# {
#     "query": "What is distance?",
#     "relevant_pages": [2]
# }
#
# =========================================================

def _retrieve_ranked_indexes(query):

    if (
        not documents
        or document_vectors is None
    ):

        return []

    query_word_vector = (
        word_vectorizer.transform(
            [query]
        )
    )

    query_char_vector = (
        char_vectorizer.transform(
            [query]
        )
    )

    query_vector = hstack(
        [
            query_word_vector,
            query_char_vector
        ]
    ).tocsr()

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    ranked_indexes = (
        similarities.argsort()[::-1]
    )

    return [
        int(index)
        for index in ranked_indexes
    ]


# =========================================================
# HIT@K
# =========================================================

def hit_at_k(
    retrieved_pages,
    relevant_pages,
    k=3
):

    retrieved = set(
        retrieved_pages[:k]
    )

    relevant = set(
        relevant_pages
    )

    return int(
        len(retrieved.intersection(
            relevant
        )) > 0
    )


# =========================================================
# PRECISION@K
# =========================================================

def precision_at_k(
    retrieved_pages,
    relevant_pages,
    k=3
):

    retrieved = retrieved_pages[:k]

    if not retrieved:

        return 0.0

    relevant = set(
        relevant_pages
    )

    hits = sum(
        1
        for page in retrieved
        if page in relevant
    )

    return hits / len(
        retrieved
    )


# =========================================================
# RECALL@K
# =========================================================

def recall_at_k(
    retrieved_pages,
    relevant_pages,
    k=3
):

    if not relevant_pages:

        return 0.0

    retrieved = set(
        retrieved_pages[:k]
    )

    relevant = set(
        relevant_pages
    )

    hits = len(
        retrieved.intersection(
            relevant
        )
    )

    return hits / len(
        relevant
    )


# =========================================================
# MRR
# =========================================================

def reciprocal_rank(
    retrieved_pages,
    relevant_pages
):

    relevant = set(
        relevant_pages
    )

    for rank, page in enumerate(
        retrieved_pages,
        start=1
    ):

        if page in relevant:

            return 1.0 / rank

    return 0.0


# =========================================================
# EVALUATE RETRIEVAL
# =========================================================

def evaluate_retrieval(
    evaluation_data,
    k=3
):

    if not evaluation_data:

        return {
            "queries": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_at_k": 0.0,
            "mrr": 0.0
        }

    precision_scores = []

    recall_scores = []

    hit_scores = []

    mrr_scores = []

    details = []

    for item in evaluation_data:

        query = item.get(
            "query",
            ""
        )

        relevant_pages = item.get(
            "relevant_pages",
            []
        )

        if not query:

            continue

        ranked_indexes = (
            _retrieve_ranked_indexes(
                query
            )
        )

        retrieved_pages = [
            documents[index]["page"]
            for index in ranked_indexes
        ]

        p = precision_at_k(
            retrieved_pages,
            relevant_pages,
            k
        )

        r = recall_at_k(
            retrieved_pages,
            relevant_pages,
            k
        )

        h = hit_at_k(
            retrieved_pages,
            relevant_pages,
            k
        )

        m = reciprocal_rank(
            retrieved_pages,
            relevant_pages
        )

        precision_scores.append(p)

        recall_scores.append(r)

        hit_scores.append(h)

        mrr_scores.append(m)

        details.append(
            {
                "query": query,

                "relevant_pages":
                    relevant_pages,

                "retrieved_pages":
                    retrieved_pages[:k],

                "precision_at_k":
                    round(p, 4),

                "recall_at_k":
                    round(r, 4),

                "hit_at_k":
                    h,

                "reciprocal_rank":
                    round(m, 4)
            }
        )

    count = len(
        precision_scores
    )

    if count == 0:

        return {
            "queries": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "details": []
        }

    return {
        "queries": count,

        "k": k,

        "precision_at_k": round(
            sum(precision_scores)
            / count,
            4
        ),

        "recall_at_k": round(
            sum(recall_scores)
            / count,
            4
        ),

        "hit_at_k": round(
            sum(hit_scores)
            / count,
            4
        ),

        "mrr": round(
            sum(mrr_scores)
            / count,
            4
        ),

        "details": details
    }


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
    # SAFE HISTORY
    # =====================================================

    def _get_safe_history(self):

        safe_history = []

        for message in self.conversation_history[
            -MAX_HISTORY_MESSAGES:
        ]:

            content = message.get(
                "content",
                ""
            )

            if isinstance(
                content,
                str
            ):

                content = content[
                    :MAX_HISTORY_MESSAGE_CHARS
                ]

            safe_history.append(
                {
                    "role": message.get(
                        "role"
                    ),

                    "content": content
                }
            )

        return safe_history


    # =====================================================
    # RUN AGENT
    # =====================================================

    def run(self, goal):

        self.last_activity = []

        # =================================================
        # SYSTEM MESSAGE
        # =================================================

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

                    "For document questions, use "
                    "document_search before answering. "

                    "When document_search is used, "
                    "answer using the retrieved "
                    "document content. "

                    "Do not invent document facts. "

                    "Mention retrieved page numbers "
                    "when useful or requested. "

                    "If the retrieved content is not "
                    "sufficient, clearly state that "
                    "the information could not be "
                    "found in the retrieved document. "

                    "When multiple document sections "
                    "are retrieved, combine them "
                    "carefully. "

                    "When web_search is used, base "
                    "the answer on returned results. "

                    "Include a Sources section when "
                    "web_search is used. "

                    "Never invent URLs. "

                    "Use previous conversation context "
                    "only when relevant. "

                    "Keep answers concise and directly "
                    "answer the user's question."
                )
            }
        ]

        # =================================================
        # ADD SAFE MEMORY
        # =================================================

        messages.extend(
            self._get_safe_history()
        )

        # =================================================
        # CURRENT REQUEST
        # =================================================

        messages.append(
            {
                "role": "user",
                "content": goal[:3000]
            }
        )

        # =================================================
        # MULTI-STEP LOOP
        # =================================================

        for iteration in range(
            MAX_AGENT_ITERATIONS
        ):

            try:

                # -------------------------------------------------
                # GROQ REQUEST
                # -------------------------------------------------

                response = (
                    groq_client.chat.completions.create(

                        model="openai/gpt-oss-20b",

                        messages=messages,

                        tools=tools,

                        tool_choice="auto",

                        parallel_tool_calls=False,

                        temperature=0.2,

                        max_tokens=1200
                    )
                )

            except Exception as error:

                error_text = str(error)

                # -------------------------------------------------
                # FRIENDLY TPM ERROR
                # -------------------------------------------------

                if (
                    "413" in error_text
                    or "tokens per minute"
                    in error_text.lower()
                    or "rate_limit" in error_text.lower()
                ):

                    return (
                        "The request was too large "
                        "for the current model limit. "
                        "Please try the question again "
                        "with a shorter query."
                    )

                return (
                    f"AI request error: {error_text}"
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

                        "content": goal[
                            :MAX_HISTORY_MESSAGE_CHARS
                        ]
                    }
                )

                self.conversation_history.append(
                    {
                        "role": "assistant",

                        "content": final_answer[
                            :MAX_HISTORY_MESSAGE_CHARS
                        ]
                    }
                )

                # -------------------------------------------------
                # KEEP MEMORY SMALL
                # -------------------------------------------------

                self.conversation_history = (
                    self.conversation_history[
                        -MAX_HISTORY_MESSAGES:
                    ]
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
                    json.JSONDecodeError,
                    TypeError
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
                # SAFETY LIMIT TOOL RESULT
                # =================================================

                if isinstance(
                    result,
                    str
                ):

                    result = result[
                        :8000
                    ]

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
            "number of tool steps. "
            "Please try a more focused question."
        )


# =========================================================
# AGENT INSTANCE
# =========================================================

agent = ResearchAgent()


















    

        
