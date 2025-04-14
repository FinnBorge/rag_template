import logging
from typing import Optional, List, Union

from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_bench.routers.api_v1.chat.chat_service import ChatService
from rag_bench.routers.api_v1.chat.types import ChatMetadata, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

class ChatMessage(BaseModel):
    """A message in a chat conversation."""
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    """Simple request format for direct query."""
    query: str
    conversation_id: Optional[str] = None
    metadata: Optional[dict] = None  # Accept raw dict instead of ChatMetadata
    stream: bool = False

class ChatRequest(BaseModel):
    """Request for chat completion with messages array."""
    messages: List[ChatMessage]
    stream: bool = Field(default=False)
    include_document_text: bool = Field(default=False)
    metadata: Optional[ChatMetadata] = Field(default_factory=ChatMetadata)
    conversation_id: Optional[str] = None

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
    request: Union[ChatRequest, ChatCompletionRequest],
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """Process a chat request and return a response."""
    logger.info(f"Received chat completion request type: {type(request).__name__}")
    # Log the raw request data for debugging
    if hasattr(request, 'model_dump'):
        logger.info(f"Request data: {request.model_dump()}")
    else:
        logger.info(f"Request data: {request}")
    
    # Handle both request formats
    if hasattr(request, 'query'):
        # Simple query format
        query = request.query
        # Convert dict metadata to ChatMetadata if needed
        if request.metadata and isinstance(request.metadata, dict):
            try:
                metadata = ChatMetadata(**request.metadata)
            except Exception as e:
                logger.error(f"Error converting metadata: {e}")
                # Use default metadata if conversion fails
                metadata = ChatMetadata()
        else:
            metadata = request.metadata or ChatMetadata()
        
        conversation_id = request.conversation_id
        include_document_text = getattr(metadata, 'include_document_text', False)
        stream = request.stream
    elif hasattr(request, 'messages'):
        # Messages array format
        if not request.messages:
            raise ValueError("No messages provided")
        
        # Get the last user message
        last_message = request.messages[-1]
        if last_message.role != "user":
            raise ValueError("Last message must be from the user")
        
        query = last_message.content
        metadata = request.metadata or ChatMetadata()
        conversation_id = request.conversation_id
        include_document_text = getattr(request, 'include_document_text', False)
        stream = request.stream
    else:
        raise ValueError("Invalid request format")
    
    logger.info(f"Processing query: {query}")
    
    # Store conversation_id in metadata if it's not already there
    if conversation_id and not metadata.conversation_id:
        metadata.conversation_id = conversation_id

    if stream:
        # Handle streaming response
        return StreamingResponse(
            chat_service.stream_response(
                query=query,
                metadata=metadata,
                include_document_text=include_document_text
            ),
            media_type="text/event-stream"
        )
    else:
        # Handle regular response
        return await chat_service.generate_response(
            query=query,
            metadata=metadata,
            include_document_text=include_document_text
        )
