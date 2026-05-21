"""
Chat API router for RAG completions.
"""
import logging
from typing import Optional, List, Union, Literal

from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from rag_bench.routers.api_v1.chat.chat_service import ChatService
from rag_bench.routers.api_v1.chat.types import ChatMetadata, ChatResponse


logger = logging.getLogger(__name__)

# Input validation constants
MAX_QUERY_LENGTH = 10000  # Characters
MAX_MESSAGES = 100
ALLOWED_ROLES = {"user", "assistant", "system"}

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


class ChatMessage(BaseModel):
    """A message in a chat conversation."""
    role: Literal["user", "assistant", "system"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Message content exceeds maximum length of {MAX_QUERY_LENGTH}")
        return v


class ChatCompletionRequest(BaseModel):
    """Simple request format for direct query."""
    query: str
    conversation_id: Optional[str] = None
    metadata: Optional[dict] = None
    stream: bool = False

    @field_validator("query")
    @classmethod
    def validate_query_length(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH}")
        return v


class ChatRequest(BaseModel):
    """Request for chat completion with messages array."""
    messages: List[ChatMessage]
    stream: bool = Field(default=False)
    include_document_text: bool = Field(default=False)
    metadata: Optional[ChatMetadata] = Field(default_factory=ChatMetadata)
    conversation_id: Optional[str] = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        if not v:
            raise ValueError("Messages list cannot be empty")
        if len(v) > MAX_MESSAGES:
            raise ValueError(f"Too many messages (max {MAX_MESSAGES})")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {"role": "user", "content": "What is diabetes?"}
                    ],
                    "stream": False,
                    "include_document_text": False,
                }
            ]
        }
    }


def get_chat_service(request: Request) -> ChatService:
    """Get the chat service from the injector."""
    return request.state.injector.get(ChatService)


@router.post("/completions", response_model=ChatResponse)
async def chat_completion(
    request: Union[ChatRequest, ChatCompletionRequest],
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Process a chat request and return a response."""
    logger.info(f"Received chat completion request type: {type(request).__name__}")

    # Handle both request formats
    if isinstance(request, ChatCompletionRequest):
        query = request.query
        metadata = _parse_metadata(request.metadata)
        conversation_id = request.conversation_id
        include_document_text = getattr(metadata, "include_document_text", False)
        stream = request.stream

    elif isinstance(request, ChatRequest):
        # Validate last message is from user
        last_message = request.messages[-1]
        if last_message.role != "user":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Last message must be from the user",
            )

        query = last_message.content
        metadata = request.metadata or ChatMetadata()
        conversation_id = request.conversation_id
        include_document_text = request.include_document_text
        stream = request.stream

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request format",
        )

    logger.info(f"Processing query (length={len(query)})")

    # Store conversation_id in metadata if not already there
    if conversation_id and not metadata.conversation_id:
        metadata = ChatMetadata(
            **{**metadata.model_dump(), "conversation_id": conversation_id}
        )

    if stream:
        return StreamingResponse(
            chat_service.stream_response(
                query=query,
                metadata=metadata,
                include_document_text=include_document_text,
            ),
            media_type="text/event-stream",
        )

    return await chat_service.generate_response(
        query=query,
        metadata=metadata,
        include_document_text=include_document_text,
    )


def _parse_metadata(metadata: Optional[dict]) -> ChatMetadata:
    """Parse metadata dict to ChatMetadata, with error handling."""
    if not metadata:
        return ChatMetadata()

    try:
        return ChatMetadata(**metadata)
    except Exception as e:
        logger.warning(f"Error parsing metadata, using defaults: {e}")
        return ChatMetadata()
