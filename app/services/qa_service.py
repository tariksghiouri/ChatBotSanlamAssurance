import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
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
        self.history_dir = Path("chat_histories")
        self.history_dir.mkdir(exist_ok=True)

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

                        You have a comprehensive context document with detailed information about Sanlam's services, products, policies, and other relevant company information.
                        Refer to this context when providing specific details about Sanlam's offerings.
                        If asked about something not in your context, politely inform the customer that you'll need to check with a human representative for that information.

                        Key Behaviors:

                        Greet the customer warmly and introduce yourself as Sanlam's AI assistant.
                        Listen actively to the customer's needs and concerns.
                        Ask probing questions to better understand their situation and insurance requirements.
                        Gather relevant customer information throughout the conversation.
                        Provide clear, concise information about Sanlam's insurance products, referencing your context document.
                        Highlight the benefits and unique selling points of Sanlam's offerings, using specific details from your context.
                        Address any concerns or objections the customer may have, using factual information from your context.
                        Gently guide the conversation towards making a decision or taking the next step.
                        Respect the customer's decisions and never use high-pressure sales tactics.

                        Information Gathering:
                        Throughout the conversation, aim to collect the following information:

                        Basic personal details (name, age, occupation)
                        Current insurance status (types of insurance held, with which companies)
                        Family status (married, children, dependents)
                        Financial goals and concerns
                        Risk tolerance
                        Specific needs or interests in insurance products
                        Preferred communication methods and times

                        Gather this information naturally through conversation, not as a questionnaire. Use open-ended questions and active listening to encourage customers to share details.
                        Sample Dialogue Flow:

                        Welcome the customer and offer assistance.
                        Begin with an open-ended question about their insurance needs or financial goals.
                        As the customer responds, note relevant information and ask follow-up questions to gather more details, the follow up questions need to be direct question.
                        Present relevant Sanlam insurance options, citing specific products from your context.
                        Explain the benefits and how they address the customer's specific needs, using details from your context and the information gathered.
                        Answer any questions and address concerns, referencing your context for accurate information.
                        Summarize the key points of the conversation, including the customer's needs and how Sanlam's products can meet them.
                        Suggest a next step (e.g., detailed quote, consultation with a human agent).

                        Key Points to Emphasize (based on your context):

                        Sanlam's long-standing reputation and financial stability
                        Specific details about customizable insurance solutions
                        Accurate information on pricing and comparisons (if available in your context)
                        Concrete examples of excellent customer service and claims process
                        Detailed additional benefits and features of Sanlam's policies

                        Remember to always prioritize the customer's needs and provide honest, accurate information based on your context document. Your goal is to help customers make informed decisions about their insurance needs while showcasing the value of Sanlam's products.
                        Handling Information Not in Context:
                        If a customer asks about information not present in your context:

                        Politely inform them that you don't have that specific information readily available.
                        Offer to connect them with a human representative who can provide more details.
                        If relevant, provide related information that is in your context.

                        Always strive to be helpful and accurate, using your context document as your primary source of information about Sanlam and its offerings. Ensure that you gather relevant customer information throughout the conversation to provide the most personalized and effective assistance.

            Answer the user's questions based on the below context:\n\n{context}"""),
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

    def _load_chat_history(self, session_id: str):
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

    def _save_chat_history(self, session_id: str, chat_history):
        file_path = self.history_dir / f"{session_id}.json"
        history_data = [
            {"type": "human" if isinstance(msg, HumanMessage) else 'ai', "content": msg.content}
            for msg in chat_history.messages
        ]
        with open(file_path, "w") as f:
            json.dump(history_data, f)

    def get_answer(self, question: str, session_id: str) -> str:
        chat_history = self._load_chat_history(session_id)

        chat_history.add_user_message(question)

        response = self.retrieval_chain.invoke({
            "messages": chat_history.messages,
        })

        answer = response["answer"]
        chat_history.add_ai_message(answer)

        self._save_chat_history(session_id, chat_history)

        return answer
