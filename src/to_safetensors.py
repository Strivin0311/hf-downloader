import os
from dotenv import load_dotenv
load_dotenv(".env")

model_root = os.getenv('HF_MODEL_ROOT')

model_type = input("Please choose the one right model types from {clm, mlm, seq2seq}:\n")
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
model_load_class = {
    'mlm': AutoModelForMaskedLM,
    'clm': AutoModelForCausalLM,
    'seq2seq': AutoModelForSeq2SeqLM,
}[model_type]

model_dir = input("Please input the model base dir:\n")
model_name = input("Please input the model name:\n")
model_path = os.path.join(model_root, model_dir, model_name)

model = model_load_class.from_pretrained(model_path)
model.save_pretrained(model_path)