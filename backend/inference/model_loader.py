import yaml
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from code.inference.chat_engine import generate_response

def load_train_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_tokenizer(config_path: str):
    config = load_train_config(config_path)
    base_model_path = config["model_name_or_path"]
    lora_path = config["output_dir"]
    print("开始加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    print("开始加载 base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("开始加载 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    print("开始合并 LoRA...")
    model = model.merge_and_unload()
    model.eval()
    print("模型加载完成")

    return model, tokenizer

