import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from app.core.config import settings

class QAService:
    def __init__(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
        self.chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)
        self.embed = OpenAIEmbeddings()
        self.conn_uri = settings.POSTGRES_URI
        self.vector_db = self._initialize_vector_db()
        self.retriever = self.vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 2})
        self.question_answering_prompt = self._create_qa_prompt()
        self.document_chain = create_stuff_documents_chain(self.chat, self.question_answering_prompt)
        self.retrieval_chain = self._create_retrieval_chain()
        self.chat_histories = {}  # Dictionary to store chat histories for different sessions

    def _initialize_vector_db(self):
        try:
            vector_db = PGVector.from_existing_index(
                collection_name="sanlam-6bg5a55aeebc4f6983ed3ef9377bad08",
                embedding=self.embed,
                connection_string=self.conn_uri,
                use_jsonb=True
            )
        except Exception as e:
            print(f"Error initializing vector database: {e}")
            print("Please check your connection string and ensure the database is properly set up.")
            raise
        return vector_db

    def _create_qa_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """
                Sanlam Insurance Assistant Prompt
                You are an AI assistant for Sanlam, a reputable insurance company. Your role is to engage with potential customers, understand their needs, guide them towards suitable insurance products, and gather relevant information about them. You have access to a context document containing detailed information about Sanlam's services, products, and policies. Always maintain a professional, friendly, and helpful demeanor.
                Access to Context:

                Answer the user's questions based on the below context:\n\n{context}
            """),
            MessagesPlaceholder(variable_name="messages"),
        ])

    def _create_retrieval_chain(self):
        def parse_retriever_input(params):
            return params["messages"][-1].content

        return RunnablePassthrough.assign(
            context=parse_retriever_input | self.retriever,
        ).assign(
            answer=self.document_chain,
        )

    def get_answer(self, question: str) -> str:
        chat_history = ChatMessageHistory()

        """
        Get an answer to a question using document retrieval and chat history.

        Args:
        question (str): The user's question.

        Returns:
        str: The AI's response.
        """
        chat_history.add_user_message(question)

        response = self.retrieval_chain.invoke({
            "messages": chat_history.messages,
        })

        answer = response["answer"]
        chat_history.add_ai_message(answer)

        return answer
