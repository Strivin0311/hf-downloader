import os
import argparse
import json

from dotenv import load_dotenv
load_dotenv(".env")

from utils import *

model_root = os.getenv('HF_MODEL_ROOT')
if model_root is None: raise ValueError("HF_MODEL_ROOT is not set")

hf_mirror = os.getenv('HF_ENDPOINT') # could be None
if hf_mirror is None: hf_mirror = "https://huggingface.co" # default huggingface official cite


def set_available_gpus(gpus_list):
    if gpus_list is None: return
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,gpus_list))

def load_model_and_tokenizer(model_path, model_type, num_gpus=1):
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM

    model_load_class = {
        'mlm': AutoModelForMaskedLM,
        'clm': AutoModelForCausalLM,
        'seq2seq': AutoModelForSeq2SeqLM,
    }[model_type]

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"The tokenizer meta information is shown as below: \n{tokenizer}\n")
    if num_gpus > 1:
        model = model_load_class.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
    else: 
        model = model_load_class.from_pretrained(model_path, trust_remote_code=True)
        if num_gpus == 1: model = model.cuda()

    print(f"The model structure is shown as below: \n{model}\n")

    return model, tokenizer

def print_model_info(model, model_path):
    print("="*50, '\n', f"The local load path of this model is: {model_path}")
    param_size = get_model_params(model, format=True)
    print("="*50, '\n', f"The parameter size of this model is: {param_size}")
    disk_size = get_dir_size(model_path, format=True)
    print("="*50, '\n', f"The disk size of this model is: {disk_size}")
    mem_size_dict = get_mem_size(format=True)
    print("="*50, '\n', f"The mem size of this model is: \n{info_dict(mem_size_dict)}")
    context_len = find_context_len(model_path)
    print("="*50, '\n',f"The maximum training context length of this model is: {context_len}")
    
    # FIXME: for statistics in our own table
    print("="*50, '\n', f"Summary:\n |{get_model_name_from_path(model_path)}|{get_model_base_from_path(model_path, model_root)}|{retrieve_highest(param_size, precision=1)}|{retrieve_highest(disk_size, precision=1)}|{retrieve_highest(mem_size_dict['sum'], precision=1)}|{format_context_len(context_len) if context_len != 'UNKNOWN' else context_len}|[here]('{model_path}')|[here]('{hf_mirror}')|")

def text_generation(model, tokenizer, num_gpus):
    # test basic text generation task with model.generation()
    # NOTE: thus the model should be causal language model loaded by AutoModelForCausalLM or the specific class
    default_input_text = "If I am iron man, I will "
    
    while True:
        print("="*50)
        input_text = input(f"Input your text to be complete, or keep it empty to use the default text (type 'quit' to exit): '{default_input_text}':\n")
        if input_text.lower() == 'quit': break
        if input_text == "": input_text = default_input_text

        if num_gpus >= 1:
            inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
        else: inputs = tokenizer([input_text], return_tensors="pt")
        print(f"The tokenized inputs for input text: '{input_text}' is: \n{inputs}")
        outputs = model.generate(**inputs, 
                                max_new_tokens=50, 
                                pad_token_id=tokenizer.eos_token_id # for open-ending generation
                                )
        print(f"The outputs from the model generation is: \n{outputs}")
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"The final output text is: \n{output_text}")
        print("="*50+"\n")

def chatbot(model, tokenizer, num_gpus):
    # test chatbot task with model.chat()
    # NOTE: the model should support the api chat with history mechanism in it like chatglm
    default_opening = "Hi, my name is Mike."
    
    history = None
    while True:
        print("="*50)
        query = input(f"Chat with the bot, or keep it empty to use the default opening and start a new turn of chatting (type 'quit' to exit): '{default_opening}':\n")
        if query.lower() == 'quit': break
        if query == "": 
            query = default_opening
            history = None

        response, history = model.chat(tokenizer, query, history)
        print(f"The response from the chatbot is: \n{response}")
        print("="*50+"\n")

def main(args):
    model_path = os.path.join(model_root, args.model_path)

    set_available_gpus(args.devices)

    model, tokenizer = load_model_and_tokenizer(model_path, args.model_type, len(args.devices))

    print_model_info(model, model_path)

    task_func_map = {
        'text_generation': text_generation,
        'chatbot': chatbot,
    }

    task_func_map[args.task](model, tokenizer, len(args.devices))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load one downloaded model to get its basic information and test it on some basic tasks")
    parser.add_argument("--model_path", type=str, required=True, help="The relative path to the downloaded model under the model root set in the '.env'")
    parser.add_argument("--model_type", type=str, default="clm", choices=['clm', 'mlm', 'seq2seq'], 
                        help="The type of the model, default is 'clm' for causal language models like GPT, while 'mlm' is for maksed language models like BERT and 'seq2seq' is for sequence-to-sequence models like T5, BART")

    parser.add_argument("--devices", type=str, nargs='*', default='0', 
                        help="The gpu devices list to use, default is '0', and if you have multiple gpus, you can list them after the argument split with a space like '0 1 3 6'")
    
    parser.add_argument("--task", type=str, default="text_generation", 
                        choices=['text_generation', 'chatbot'], 
                        help="The available task to use the model, default is 'text_generation'")

    args = parser.parse_args()

    main(args)

