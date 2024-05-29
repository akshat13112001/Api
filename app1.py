from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

class InputData(BaseModel):
    prompt: str

class OutputData(BaseModel):
    response: str

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")

@app.post("/generate", response_model=OutputData)
def generate(request: Request, input_data: InputData):
    prompt = [{'role': 'user', 'content': input_data.prompt}]
    inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=False, return_tensors='pt')
    prompt_length = inputs[0].shape[0]
    tokens = model.generate(inputs.to(model.device), max_new_tokens=1024, temperature=0.8, do_sample=True)
    response = tokenizer.decode(tokens[0][prompt_length:], skip_special_tokens=True)
    return OutputData(response=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)