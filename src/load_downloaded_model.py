import os
import argparse
import json

from dotenv import load_dotenv
load_dotenv(".env")

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_model_params

model_root = os.getenv('HF_MODEL_ROOT')
if model_root is None: raise ValueError("HF_MODEL_ROOT is not set")


def main(args):

    model_path = os.path.join(model_root, args.model_path)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"The tokenizer meta information is shown as below: \n{tokenizer}\n")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print(f"The model structure is shown as below: \n{model}\n")
    print(f"whose the number of parameters is : {get_model_params(model)}")

    # test basic open-ending generation
    input_text = "If you have a car, you would like to"
    inputs = tokenizer([input_text], return_tensors="pt")
    print(f"The tokenized inputs for input text: '{input_text}' is: \n{inputs}")
    outputs = model.generate(**inputs, 
                             max_new_tokens=20, 
                             pad_token_id=tokenizer.eos_token_id # for open-ending generation
                            )
    print(f"The outputs from the model generation is: \n{outputs}")
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"The final output text is: \n{output_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load one downloaded model to test if the downloaded files are enough for basic usage")
    parser.add_argument("--model_path", type=str, required=True, help="The relative path to the downloaded model under the model root set in the '.env'")

    args = parser.parse_args()

    main(args)

