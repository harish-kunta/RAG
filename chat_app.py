"""Local web chat UI for the learning RAG pipeline."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag
import langchain_rag
from conversation_store import ConversationStore


ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"
logger = logging.getLogger(__name__)
conversation_store = ConversationStore(ROOT / "data" / "chat_history.sqlite3")


@asynccontextmanager
async def lifespan(_: FastAPI):
    conversation_store.initialize()
    yield

app = FastAPI(
    title="RAG Chat",
    description="A local chat interface for the learning RAG pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=36)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    """Liveness check; it does not make a paid model request."""

    try:
        vector_index_present = langchain_rag.index_directory().exists()
    except RuntimeError:
        vector_index_present = False
    return {"status": "ok", "vector_index_present": vector_index_present}


@app.get("/api/conversations")
def list_conversations() -> dict[str, object]:
    return {"conversations": conversation_store.list_conversations()}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, object]:
    conversation = conversation_store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> Response:
    if not conversation_store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return Response(status_code=204)


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Enter a question first.")

    conversation = None
    if request.conversation_id:
        conversation = conversation_store.get_conversation(
            request.conversation_id,
            message_limit=12,
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    history = conversation["messages"] if conversation else []

    # A prior question helps with simple follow-ups such as “what about returns?”.
    # This is intentionally a small starter technique; production systems usually
    # add a measured query-rewriting step for conversational retrieval.
    prior_user_questions = [message["content"] for message in history if message["role"] == "user"]
    retrieval_query = "\n".join([*prior_user_questions[-1:], question])

    try:
        vector_store = langchain_rag.load_vector_store(rag.DATA_DIR)
        results = langchain_rag.search_hybrid(
            retrieval_query,
            vector_store,
            rag.DATA_DIR,
            top_k=6,
        )
        context, _, _ = rag.pack_context(results, max_chars=5000)
        answer = langchain_rag.generate_answer(
            question,
            context,
            [(message["role"], message["content"]) for message in history],
        )
    except RuntimeError as exc:
        # Configuration and stale-index errors are actionable for a local learner.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=502,
            detail="The retrieval or model service failed. Check the server terminal and try again.",
        ) from exc

    sources = []
    for score, chunk in results:
        item: dict[str, object] = {
            "label": f"{rag.source_label(chunk)}, chunk {chunk.number}",
            "source": chunk.source,
            "chunk": chunk.number,
            "excerpt": chunk.text,
            "rank_score": round(score, 5),
        }
        url = chunk.metadata.get("url")
        if url:
            item["url"] = url
        sources.append(item)

    saved_conversation_id = conversation_store.append_turn(
        question,
        answer,
        sources,
        conversation_id=request.conversation_id,
    )
    return {
        "answer": answer,
        "sources": sources,
        "conversation_id": saved_conversation_id,
    }
