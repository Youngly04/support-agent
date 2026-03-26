from fastapi import FastAPI

from code.inference.model_loader import load_model_and_tokenizer
from code.api.chat_api import router, set_model_and_tokenizer
from fastapi.middleware.cors import CORSMiddleware


CONFIG_PATH = "xxx/train_lora.yaml"

app = FastAPI(title="Qwen LoRA Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model, tokenizer = load_model_and_tokenizer(CONFIG_PATH)
set_model_and_tokenizer(model, tokenizer)
#post
#get
#delete
#put/patch
app.include_router(router)