def generate_response(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(

        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    output_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response.strip()