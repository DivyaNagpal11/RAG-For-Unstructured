from typing import List, Dict, Optional, Any
from langchain_core.language_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_community.llms import Ollama


class OllamaChatLLM(SimpleChatModel):
    access: Dict
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ollama_model = Ollama(
            model=self.access.get("model", "mistral"),
            temperature=0.1,
            num_predict=1024,
            top_p=0.95
        )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {"access": self.access}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "Ollama"

    def _call(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,) -> str:
        """Process chat messages and return a response."""
        
        # Format messages for the model
        formatted_messages = []
        for message in messages:
            role = "user" if message.type == "human" else message.type
            formatted_messages.append({"role": role, "content": message.content})
        
        # Create a combined prompt from the messages
        prompt = ""
        for msg in formatted_messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        
        prompt += "Assistant: "
        
        # Get response from the model
        response = self.ollama_model(prompt)
        return response