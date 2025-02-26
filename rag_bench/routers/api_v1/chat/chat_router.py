from typing import Optional, List

from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_bench.routers.api_v1.chat.chat_service import ChatService
from rag_bench.routers.api_v1.chat.types import ChatMetadata, ChatResponse


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


class ChatMessage(BaseModel):
    """A message in a chat conversation."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request for chat completion."""
    messages: List[ChatMessage]
    stream: bool = Field(default=False)
    include_document_text: bool = Field(default=False)
    metadata: Optional[ChatMetadata] = Field(default_factory=ChatMetadata)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is diabetes?"
                        }
                    ],
                    "stream": False,
                    "include_document_text": False,
                    "metadata": {
                        "user_id": 123
                    }
                }
            ]
        }
    }


def get_chat_service(request: Request) -> ChatService:
    """Get the chat service from the injector."""
    return request.state.injector.get(ChatService)


@router.post(
    "/completions",
    response_model=ChatResponse
)
async def chat_completion(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """Process a chat request and return a response."""
    if not request.messages:
        raise ValueError("No messages provided")
    
    # Get the last user message
    last_message = request.messages[-1]
    if last_message.role != "user":
        raise ValueError("Last message must be from the user")
    
    # Set up metadata
    metadata = request.metadata or ChatMetadata()
    
    if request.stream:
        # Handle streaming response
        return StreamingResponse(
            chat_service.stream_response(
                query=last_message.content,
                metadata=metadata,
                include_document_text=request.include_document_text
            ),
            media_type="text/event-stream"
        )
    else:
        # Handle regular response
        return await chat_service.generate_response(
            query=last_message.content,
            metadata=metadata,
            include_document_text=request.include_document_text
        )