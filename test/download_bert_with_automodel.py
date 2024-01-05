import os
from dotenv import load_dotenv
load_dotenv(".env")

model_root = os.getenv('HF_MODEL_ROOT')

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, BertModel, BertForMaskedLM, BertForSequenceClassification


model_name = "bert-base-uncased"
save_path = os.path.join(model_root, "bert", model_name)


### NOTE 1: the 'from_pretrained' from AutoModel below will load the config.json an model.safetensors for BertForMaskedLM model 
# since bert-base-uncased is a pretrained model using MLM objective
# but it will only instantiate the BertModel model and when you 'save_pretrained', 
# it will only save for BertModel, i.e. not only save the BertModel-like config.json and the weights only for BertModel

# model = AutoModel.from_pretrained(model_name)
# print(model)
# model.save_pretrained(save_path)

### NOTE 2: the 'from_pretrained' from AutoModelForMaskedLM below will also load all the right config.json and model.safetensors for BertForMaskedLM model
# and it will instantiate BertForCausalLM and 'save_pretrained' will save all the weights correctly, since bert-base-uncased is exactly the AutoModelForMaskedLM 

# model = AutoModelForMaskedLM.from_pretrained(model_name)
# print(model)
# model.save_pretrained(save_path)

### NOTE 3: the same as above, except we point out the concrete model class

# model = BertForMaskedLM.from_pretrained(model_name)
# print(model)
# model.save_pretrained(save_path)

### NOTE 4: this is the subclass of BertForMaskedLM, but since this repo contains only the weights for BertForMaskedLM,
# so only the model structure of BertForSequenceClassification will be instantiated 
# and the weights specific for BertForSequenceClassification will be randomly instantiated

# model = BertForSequenceClassification.from_pretrained(model_name)
# print(model)
# model.save_pretrained(save_path)



tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
print(model)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)



