import os
from dotenv import load_dotenv
load_dotenv(".env")

model_root = os.getenv('HF_MODEL_ROOT')

from transformers import AutoTokenizer, AutoModel

model_path = os.path.join(model_root, "chatglm/THUDM/chatglm3-6b-base")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

inputs = tokenizer(["今天天气真不错"], return_tensors="pt").to('cuda')
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0].tolist()))
