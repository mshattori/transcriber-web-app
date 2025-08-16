"""
Chat handler - manages chat interactions with transcript context.

Separates chat business logic from UI event handlers.
"""

from typing import List, Dict, Any, Tuple

from llm import chat_with_context, chat_completion


class ChatHandler:
    """Real chat handler."""
    
    def __init__(self):
        self.chat_history: List[Dict[str, str]] = []
        self.current_context: str = ""
    
    def set_context(self, context_text: str):
        """
        Set the context text for chat interactions.
        
        Args:
            context_text: Transcript text to use as context
        """
        self.current_context = context_text
    
    def handle_message(
        self,
        message: str,
        history: List[Dict[str, str]], 
        settings: Dict[str, Any]
    ) -> Tuple[List[Dict[str, str]], str]:
        """
        Handle chat message with context injection.
        
        Args:
            message: User message
            history: Current chat history in Gradio messages format
            settings: App settings including API key and model
            
        Returns:
            Tuple of (updated_history, empty_input)
        """
        if not message.strip():
            return history, ""
        
        if not settings.get("api_key"):
            raise ValueError("Please set your OpenAI API key in settings")
        
        # Use transcript as context if available
        context_text = self.current_context
        
        if context_text:
            response = chat_with_context(
                api_key=settings["api_key"],
                model=settings["language_model"],
                question=message,
                context_text=context_text,
                system_message=settings.get("system_message", ""),
                temperature=0.7
            )
        else:
            response, _ = chat_completion(
                api_key=settings["api_key"],
                model=settings["language_model"],
                message=message,
                system_message=settings.get("system_message", ""),
                temperature=0.7
            )
        
        # Update history in Gradio messages format
        new_history = history.copy() if history else []
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": response})
        
        return new_history, ""
    
    def clear_history(self) -> List[Dict[str, str]]:
        """
        Clear chat history.
        
        Returns:
            Empty chat history
        """
        self.chat_history = []
        return []


class MockChatHandler:
    """Mock chat handler for UI testing."""
    
    def __init__(self):
        self.chat_history: List[Dict[str, str]] = []
        self.current_context: str = "Mock transcript context"
    
    def set_context(self, context_text: str):
        """Mock context setting."""
        self.current_context = context_text
    
    def handle_message(
        self,
        message: str,
        history: List[List[str]],
        settings: Dict[str, Any]
    ) -> Tuple[List[List[str]], str]:
        """Mock message handling - returns instant responses."""
        if not message.strip():
            return history, ""
        
        # Generate mock response based on message content
        mock_responses = {
            "hello": "Hello! I'm a mock chat assistant. How can I help you with your transcript?",
            "summary": "Here's a mock summary: This transcript contains important information about...",
            "translate": "Here's a mock translation: この文書には重要な情報が含まれています...",
            "key points": "Mock key points:\n1. First important point\n2. Second important point\n3. Third important point",
        }
        
        # Find matching response or use default
        response = None
        for key, value in mock_responses.items():
            if key.lower() in message.lower():
                response = value
                break
        
        if not response:
            response = f"Mock response to: '{message}'. This is a simulated chat response for UI testing purposes."
        
        # Add context information if available
        if self.current_context and len(self.current_context) > 50:
            response += "\n\n(This response was generated using the transcript as context)"
        
        # Update history
        history.append([message, response])
        return history, ""
    
    def clear_history(self) -> List[List[str]]:
        """Mock history clearing."""
        self.chat_history = []
        return []