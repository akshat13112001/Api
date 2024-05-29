from fastapi import FastAPI, Request
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
 prompt: str

class OutputData(BaseModel):
 response: str

model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")


@app.post("/generate", response_model=OutputData)
def generate(request: Request, input_data: InputData):
 prompt = input_data.prompt

 response = model(prompt)[0]["generated_text"]

 return OutputData(response=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)