from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "QA System API"
    OPENAI_API_KEY: str
    POSTGRES_URI: str
    COLLECTION_NAME: str

    class Config:
        env_file = ".env"

settings = Settings()