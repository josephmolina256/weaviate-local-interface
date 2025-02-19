
from fastapi import FastAPI, Request


from .interface import WeaviateInterface

app = FastAPI()

weaviate_interface = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/spawn_client")
def spawn_client(request: Request):
    global weaviate_interface
    query_params = request.query_params
    weaviate_interface = WeaviateInterface(
        generate_embeddings=query_params["generate_embeddings"],
        hf_model_name=query_params["hf_model_name"]
        )
    if query_params["generate_embeddings"] == "True":
        return {"status": "success", "message": "Client spawned! We'll take care of the embeddings."}
    else:
        return {"status": "success", "message": "Client spawned! Please provide the embeddings."}

@app.post("/embed/vectorless")
async def embed_vectorless(request: Request):
    payload = await request.json()
    input_data = payload["input_data"]
    key_to_be_embedded = payload["key_to_be_embedded"]
    collections_name = payload["collections_name"]
    result = weaviate_interface.store(input_data, key_to_be_embedded, collections_name)
    return {"status": result}

@app.post("/embed/vectorful")
async def embed_vectorful(request: Request):
    payload = await request.json()
    input_data = payload["input_data"]
    collections_name = payload["collections_name"]
    embeddings = payload["embeddings"]
    result = weaviate_interface.store(input_data, collections_name, embeddings)
    return {"status": result}


@app.get("/search/vectorless")
def search_vectorless(request: Request):
    query_params = request.query_params
    search_string = query_params["search_string"]
    collections_name = query_params["collections_name"]
    result = weaviate_interface.retrieve(search_string, collections_name)
    return {"result": result}

@app.get("/search/vectorful")
def search_vectorful(request: Request):
    query_params = request.query_params
    search_string = query_params["search_string"]
    query_embedding = query_params["query_embedding"]
    collections_name = query_params["collections_name"]
    result = weaviate_interface.retrieve(search_string, collections_name, query_embedding)
    return {"result": result}


if __name__ == "__main__":
    import uvicorn
    import signal
    import sys

    def cleanup(signum=None, frame=None):
        print("Signal received, shutting down gracefully...")
        weaviate_interface.client.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)  # Handles Ctrl+C
    signal.signal(signal.SIGTERM, cleanup)  # Handles process termination

    try:
        uvicorn.run(app, host="0.0.0.0", port=50001)
    except KeyboardInterrupt:
        cleanup()