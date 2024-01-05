import os
from dotenv import load_dotenv
load_dotenv(".env")

model_root = os.getenv('HF_MODEL_ROOT')

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM


model_path = os.path.join(model_root, "bert/bert-base-uncased")


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
print(model)

input_text = "The capital of France is [MASK]."
print(f"The input text is:\n{input_text}")
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

output_text = tokenizer.decode(predicted_token_id)
print(f"The output text is:\n{output_text}")