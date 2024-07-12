from pymongo import MongoClient
from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from .persistence_strategy import PersistenceStrategy

class MongoPersistenceStrategy(PersistenceStrategy):
    def __init__(self, connection_string: str, db_name: str, collection_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def load_chat_history(self, session_id: str) -> ChatMessageHistory:
        document = self.collection.find_one({"session_id": session_id})
        if document:
            chat_history = ChatMessageHistory()
            for msg in document['messages']:
                if msg['type'] == 'human':
                    chat_history.add_user_message(msg['content'])
                else:
                    chat_history.add_ai_message(msg['content'])
            return chat_history
        return ChatMessageHistory()

    def save_chat_history(self, session_id: str, chat_history: ChatMessageHistory):
        history_data = [
            {"type": "human" if isinstance(msg, HumanMessage) else 'ai', "content": msg.content}
            for msg in chat_history.messages
        ]
        self.collection.update_one(
            {"session_id": session_id},
            {"$set": {"messages": history_data}},
            upsert=True
        )