from fastapi import FASTAPI
import uvicorn

app = FastAPI()

@app.get('index')
def hello_world():
    return "Hello world!"

if __name__ == "__main__":
    uvicorn.run(app, port="8080", host='0.0.0.0')