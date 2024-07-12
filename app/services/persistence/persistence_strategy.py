from abc import ABC, abstractmethod
from langchain.memory import ChatMessageHistory

class PersistenceStrategy(ABC):
    @abstractmethod
    def load_chat_history(self, session_id: str) -> ChatMessageHistory:
        pass

    @abstractmethod
    def save_chat_history(self, session_id: str, chat_history: ChatMessageHistory):
        pass