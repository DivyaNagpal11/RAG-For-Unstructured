from typing import List, Dict, Optional, Any, Callable
from langchain_core.language_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_community.llms import Ollama


class OllamaChatLLMStream(SimpleChatModel):
    access: Dict
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ollama_model = Ollama(
            model=self.access.get("model", "mistral"),
            temperature=0.1,
            num_predict=1024,
            top_p=0.95,
            streaming=True
        )
        self.callback = self.access.get("callback", None)

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
        
        # Get response from the model with streaming enabled
        if self.callback:
            return self.callback(self.ollama_model.stream(prompt))
        else:
            # Collect chunks and combine for non-streaming response
            chunks = []
            for chunk in self.ollama_model.stream(prompt):
                chunks.append(chunk)
            return "".join(chunks)