import datetime
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import HTMLResponse 
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from llama_cpp import Llama
from qdrant_client import QdrantClient
from sse_starlette.sse import EventSourceResponse
from src.memory_database import MemoryDatabaseManager
from src.assistant import Assistant, StreamingWebCallbackHandler
from src.assistant.chains.chat import Chat

assistant: Assistant
memory_db: MemoryDatabaseManager
client: QdrantClient
embedding_llm: Llama

@asynccontextmanager
async def lifespan(_: FastAPI):
    global assistant, memory_db, client, embedding_llm
    assistant = Chat("./models/llama-2-7b-chat.Q4_K_M.gguf")
    embedding_llm = Llama(
        model_path="./models/mxbai-embed-large-v1-f16.gguf",
        embedding=True,
        verbose=False,
        n_batch=512,
        max_tokens=512
    )
    memory_db = MemoryDatabaseManager(db_path="chat_history.db", embedding_llm=embedding_llm)
    client = QdrantClient(path="test_embeddings")
        
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

    question = "USER: " + message.data

    # user_input_embedding = memory_db.get_local_embedding(question)
    # top_similar = memory_db.find_top_n_similar(user_input_embedding, n=1)
    # if not top_similar.empty:
    #     similar_results = "\n".join(top_similar['user_input_plus_response'].values)
    #     # print(f"Similar past interactions found:\n{similar_results}\n")
    #     # Optionally, append this to the current input for more context
    #     question += f"\nLONG MEMORY: {similar_results}"

    query_vector = embedding_llm.create_embedding(message.data)['data'][0]['embedding']
    search_results = client.search(
        collection_name="podcast",
        query_vector=query_vector,
        limit=1
    )
    #print(f"Context:\n{search_results}\n")
    context = "\n\n".join([row.payload['text'] for row in search_results])
    question = f"\n\nContext: {context}" + "User: " + message.data

    print("QUESTION: ", question)

    tasks.add_task(chain.invoke, question)

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