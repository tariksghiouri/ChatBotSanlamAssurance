from fastapi import FastAPI
from app.api.endpoints import router as qa_router
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME,docs_url="/QA_System_API_DOCS")



app.include_router(qa_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)