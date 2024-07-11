from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel
from app.services.qa_service import QAService
from app.core.security import get_api_key
import uuid

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
        question_request: QuestionRequest,
        qa_service: QAService = Depends(QAService),
        api_key: str = Depends(get_api_key),
        x_session_id: str = Header(None)
):
    if not x_session_id:
        x_session_id = str(uuid.uuid4())

    answer = qa_service.get_answer(question_request.question, x_session_id)
    return AnswerResponse(answer=answer)
