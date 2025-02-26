import logging
from typing import Optional, Dict, Any

from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from injector import inject

from rag_bench.settings.settings import Settings


logger = logging.getLogger(__name__)


class OpenAILLMComponent:
    """Component for generating text using OpenAI APIs."""
    
    @inject
    def __init__(self, settings: Settings):
        self.settings = settings
        self.openai_settings = settings.openai
        
        if not self.openai_settings:
            raise ValueError("OpenAI settings are required for OpenAILLMComponent")
            
        # Initialize client
        self.client = AsyncOpenAI(
            api_key=self.openai_settings.api_key,
            base_url=self.openai_settings.api_base
        )
        
        # Initialize LangChain client for compatibility
        self.langchain_client = ChatOpenAI(
            model=self.openai_settings.model,
            openai_api_key=self.openai_settings.api_key,
            openai_api_base=self.openai_settings.api_base
        )
    
    async def agenerate(self, template: str, **kwargs) -> str:
        """Generate text using the OpenAI API."""
        # Format the template with the provided variables
        formatted_template = template.format(**kwargs)
        
        # Make the API call
        response = await self.client.chat.completions.create(
            model=self.openai_settings.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": formatted_template}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        
        # Extract and return the generated text
        return response.choices[0].message.content


class AnthropicLLMComponent:
    """Component for generating text using Anthropic's Claude API."""
    
    @inject
    def __init__(self, settings: Settings):
        self.settings = settings
        self.anthropic_settings = settings.anthropic
        
        if not self.anthropic_settings:
            raise ValueError("Anthropic settings are required for AnthropicLLMComponent")
            
        # Import here to avoid forcing dependency if not used
        from langchain_anthropic import ChatAnthropic
        
        # Initialize LangChain client
        self.langchain_client = ChatAnthropic(
            model=self.anthropic_settings.model,
            anthropic_api_key=self.anthropic_settings.api_key
        )
    
    async def agenerate(self, template: str, **kwargs) -> str:
        """Generate text using the Anthropic API via LangChain."""
        # Format the template with the provided variables
        formatted_template = template.format(**kwargs)
        
        # Use LangChain for simplicity
        messages = [
            {"role": "human", "content": formatted_template}
        ]
        
        response = await self.langchain_client.agenerate(messages=[messages])
        
        # Extract and return the generated text
        return response.generations[0][0].text


class MockLLMComponent:
    """Mock component for testing without API calls."""
    
    @inject
    def __init__(self, settings: Settings):
        self.settings = settings
    
    async def agenerate(self, template: str, **kwargs) -> str:
        """Return a mock response for testing."""
        return "This is a mock response for testing purposes."
        
        
class LocalLLMComponent:
    """Component for generating text using locally-hosted LLM APIs."""
    
    @inject
    def __init__(self, settings: Settings):
        self.settings = settings
        self.local_settings = settings.local_llm
        
        if not self.local_settings:
            raise ValueError("Local LLM settings are required for LocalLLMComponent")
            
        # Initialize client
        try:
            from llama_cpp import Llama
            
            model_path = self.local_settings.model_path
            
            # Check if model exists
            import os
            if not os.path.exists(model_path):
                error_msg = f"Model path does not exist: {model_path}"
                logger.error(error_msg)
                logger.error("Please run 'poetry run python initialize_models.py' to download the model first.")
                logger.error("Or update settings.yaml to use mode: mock for llm if you want to proceed without a model.")
                raise FileNotFoundError(error_msg)
            
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.local_settings.context_length,
                n_gpu_layers=self.local_settings.n_gpu_layers
            )
            
            logger.info(f"Initialized local LLM with model {model_path}")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing local LLM: {e}")
            raise
    
    async def agenerate(self, template: str, **kwargs) -> str:
        """Generate text using the local LLM."""
        # Format the template with the provided variables
        formatted_template = template.format(**kwargs)
        
        # Make the API call
        try:
            response = self.model(
                formatted_template,
                max_tokens=self.local_settings.max_tokens or 1024,
                temperature=self.local_settings.temperature or 0.7,
                stop=self.local_settings.stop_sequences or ["Human:", "Assistant:"]
            )
            
            return response['choices'][0]['text']
        except Exception as e:
            logger.error(f"Error generating text with local LLM: {e}")
            return "Error generating response with local model."