import datetime
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import HTMLResponse 
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from src.assistant import Assistant, StreamingWebCallbackHandler
from src.assistant.chains.chat import Chat

"""
FastAPI server for chatbot application using Qdrant for vector-based memory retrieval.
Handles user interactions by generating embeddings, searching for relevant context, and streaming real-time responses. 
Includes endpoints for message handling, response streaming, and a web-based chat interface.

Note: Old Langchain seems terrible for prompt modification/control, leading to setup with too many globals and weird 
      grouping/pairings. Refactor with from scratch implementation if time allows or just regular RAG pipeline
      but im in too deep atm
"""

assistant: Assistant

@asynccontextmanager
async def lifespan(_: FastAPI):
    global assistant
    assistant = Chat("./models/llama-2-7b-chat.Q4_K_M.gguf")

    yield

    assistant.chains.clear()

templates = Jinja2Templates(directory="templates")
app = FastAPI(lifespan=lifespan)

@app.get('/response/{user_id}')
async def streamed_response(user_id: str):
    chain = assistant.get_chain(user_id)
    if chain is None or len(chain.callbacks) <= 0:
        return Response(status_code=422)

    handler: StreamingWebCallbackHandler = chain.callbacks[0]

    def generate():
        while True:
            while len(handler.tokens) > 0:
                token = handler.tokens.pop(0)

                yield {
                    "event": "assistant-responding",
                    "id": handler.response_id,
                    "data": token
                }

            if handler.is_responding == False:
                yield {
                    "event": "assistant-waiting",
                    "id": handler.response_id,
                    "data": 'waiting'
                }
            elif handler.response != None:
                yield {
                    "event": "assistant-response",
                    "id": handler.response_id,
                    "data": handler.get_response()
                }

    return EventSourceResponse(generate())

class Message(BaseModel):
    username: str
    data: str


@app.post('/message')
async def handle_message(message: Message, tasks: BackgroundTasks):
    chain, chain_hash = assistant.add_chain(
        key=message.username, 
        human_prefix=message.username
    )


    tasks.add_task(chain.invoke, message.data)

    return {
        'id': str(chain_hash),
        'name': message.username,
        'message': message.data,
        'timestamp': datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    }

@app.get('/', response_class=HTMLResponse)
async def chat_ui(req: Request):
    return templates.TemplateResponse('chat_ui.html', { "request": req })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")