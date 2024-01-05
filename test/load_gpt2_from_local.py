import os
import sys
sys.path.append(os.path.join(os.getcwd(), "src"))

print(sys.path)

from dotenv import load_dotenv
load_dotenv(".env")

model_root = os.getenv('HF_MODEL_ROOT')

model_size = input("Please choose one of the the model size you want (small, medium, large, xl):\n")

model_name = {
    'small': 'gpt2',
    'medium': 'gpt2-medium',
    'large': 'gpt2-large',
    'xl': 'gpt2-xl',
}[model_size]


from transformers import GPT2Tokenizer, GPT2Model, pipeline, set_seed

from utils import info_dict

model_path = os.path.join(model_root, "gpt2", model_name)

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
generator = pipeline('text-generation', model=model_path)
set_seed(42)

prompt = "Hello, I'm a language model,"

print(f"Given prompt: {prompt}\nThe sampled completions are as below:\n")

outputs = generator(prompt, 
                max_length=30, 
                num_return_sequences=3, 
                pad_token_id=tokenizer.eos_token_id
                )

for i, output in enumerate(outputs):
    print(f"Completion {i}:\n{info_dict(output)}\n")

