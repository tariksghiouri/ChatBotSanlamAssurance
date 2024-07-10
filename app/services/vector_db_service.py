from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

class VectorDBService:
    def __init__(self):
        self.embed = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        self.vector_db = self._initialize_vector_db()

    def _initialize_vector_db(self):
        return PGVector.from_existing_index(
            collection_name=settings.COLLECTION_NAME,
            embedding=self.embed,
            connection_string=settings.POSTGRES_URI,
            use_jsonb=True
        )

    def get_retriever(self):
        return self.vector_db.as_retriever(search_kwargs={"k": 4})