from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# FastAPI App
app = FastAPI()

# Load Model
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

# Define Request Body
class InputData(BaseModel):
    prompt: str

# API Route
@app.post("/generate")
async def generate_text(data: InputData):
    result = generator(data.prompt, max_length=100, num_return_sequences=1)
    return {"response": result[0]["generated_text"]}
