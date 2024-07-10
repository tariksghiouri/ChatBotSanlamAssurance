# FastAPI QA System

This project implements a question-answering system using FastAPI, leveraging a vector database for efficient document retrieval.

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <project-directory>
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Copy the `.env.example` file to `.env` and fill in the required values.

5. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`. You can access the Swagger UI documentation at `http://localhost:8000/docs`.

## Usage

Send a POST request to `/api/ask` with a JSON body containing the question:

```json
{
  "question": "Your question here"
}
```

Include the API key in the `X-API-Key` header for authentication.

## Testing

To run tests:

```
pytest
```

## License

[MIT License](LICENSE)