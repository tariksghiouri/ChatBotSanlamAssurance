import json
from pathlib import Path
from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from .persistence_strategy import PersistenceStrategy

class FilePersistenceStrategy(PersistenceStrategy):
    def __init__(self, history_dir: Path):
        self.history_dir = history_dir
        self.history_dir.mkdir(exist_ok=True)

    def load_chat_history(self, session_id: str) -> ChatMessageHistory:
        file_path = self.history_dir / f"{session_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                history_data = json.load(f)
                chat_history = ChatMessageHistory()
                for msg in history_data:
                    if msg['type'] == 'human':
                        chat_history.add_user_message(msg['content'])
                    else:
                        chat_history.add_ai_message(msg['content'])
                return chat_history
        return ChatMessageHistory()

    def save_chat_history(self, session_id: str, chat_history: ChatMessageHistory):
        file_path = self.history_dir / f"{session_id}.json"
        history_data = [
            {"type": "human" if isinstance(msg, HumanMessage) else 'ai', "content": msg.content}
            for msg in chat_history.messages
        ]
        with open(file_path, "w") as f:
            json.dump(history_data, f)