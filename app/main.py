import datetime
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from assistant import Assistant

"""
FastAPI server for chatbot application using Qdrant for vector-based memory retrieval.
Handles user interactions by generating embeddings, searching for relevant context, and streaming real-time responses. 
Includes endpoints for message handling, response streaming, and a web-based chat interface.      
"""

assistant: Assistant


@asynccontextmanager
async def lifespan(_: FastAPI):
    global assistant
    assistant = Assistant("../models/llama-2-7b-chat.Q4_K_M.gguf")

    yield  # Yielding to keep the app running and active until shutdown

    assistant.session_data.clear()  # Clearing sessions on app shutdown


templates = Jinja2Templates(directory="templates")
app = FastAPI(lifespan=lifespan)


@app.get('/response/{user_id}')
async def streamed_response(user_id: str):
    session = assistant.get_session(user_id)
    if session is None or "handler" not in session:
        return Response(status_code=422)

    handler = session["handler"]  # Get the user-specific handler


    # Streaming
    def generate():
        while True:
            while len(handler.tokens) > 0:
                token = handler.tokens.pop(0)

                yield {
                    "event": "assistant-responding",
                    "id": handler.response_id,
                    "data": token
                }

            # Yielding 'waiting' event if the assistant is not responding
            if not handler.is_responding:
                yield {
                    "event": "assistant-waiting",
                    "id": handler.response_id,
                    "data": 'waiting'
                }
            # Sending the complete response once ready
            elif handler.response is not None:
                yield {
                    "event": "assistant-response",
                    "id": handler.response_id,
                    "data": handler.get_response()
                }


    return EventSourceResponse(generate())  # Streaming response to frontend


class Message(BaseModel):
    username: str
    data: str


@app.post('/message')
async def handle_message(message: Message, tasks: BackgroundTasks):
    # Retrieve or create a session for the user
    session_data, session_hash = assistant.add_session(
        key=message.username,
        human_prefix=message.username
    )

    tasks.add_task(assistant.invoke_chain, message.username, message.data)

    # Returning metadata about the message to the frontend
    return {
        'id': str(session_hash),
        'name': message.username,
        'message': message.data,
        'timestamp': datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    }


# GET endpoint to serve the chat UI template for the frontend
# Renders the chat UI template with Jinja2, providing an interface for the user to interact with
@app.get('/', response_class=HTMLResponse)
async def chat_ui(req: Request):
    return templates.TemplateResponse('chat_ui.html', {"request": req})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
