from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn

from prediction import predict, preprocess, read_image


app = FastAPI(title='Tensorflow FastAPI Starter Pack')



@app.post('/api/predict')
async def predict_image(file: bytes = File(...)):
    
    #   Read file uploaded by user
    image = read_image(file)
    #   preprocess image
    image = preprocess(image)
    #   Make prediction
    score = predict(image)
    print(score)
    return score
if __name__ == "__main__":
    uvicorn.run(app, port="8080", host='0.0.0.0')