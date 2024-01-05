import os
from typing import List, Dict
import json


def update_config_dict(d1, d2):
    """To update the dictionary d1 with values from dictionary d2, 
    but only for keys that already exist in d1
    """
    
    d1.update({k: d2[k] for k in d1 if k in d2})
    return d1

def get_dir_size(dir_path, format=True):
    total_size = 0

    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
            
    if not format: return total_size
    
    gb_size = total_size // (1024 ** 3)
    mb_size = (total_size % (1024 ** 3)) // (1024 ** 2)
    kb_size = (total_size % (1024 ** 2)) // 1024
    size_str = f"{gb_size}GB {mb_size}MB {kb_size}KB"
    
    return size_str

def add_size_str(s1: str, s2: str) -> str:
    """size_str format: '{gb_size}GB {mb_size}MB {kb_size}KB' """
    g1, m1, k1 = [int(elem[:-2]) for elem in s1.split()]
    g2, m2, k2 = [int(elem[:-2]) for elem in s2.split()]
    
    g, m, k = g1+g2, m1+m2, k1+k2
    m += k // 1024
    k = k % 1024
    g += m // 1024
    m = m % 1024
    
    return f"{g}GB {m}MB {k}KB"
   
def get_model_params(model, format=True, only_learnable=False):
    """string format should be: '{}B {}M' """
    num_params = sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_learnable))
    if not format: return num_params
    
    b_size = num_params // (1000 ** 3)
    m_size = (num_params % (1000 ** 3)) // (1000 ** 2)
    return f"{b_size}B {m_size}M" 

def get_proxies_dict(proxies: str) -> Dict[str, str]:
    if proxies is None: return None

    proxies_dict = {}
    if proxies == "": return proxies_dict

    for proxy in proxies.split(','):
        k, v = proxy.split('://')
        proxies_dict[k.strip()] = v.strip()
    
    return proxies_dict

def get_order_str(order_num: int) -> str:
    orderstr_dict = {-1: " ", 0: " 1st ", 1: " 2nd ", 2: " 3rd "}
    order_str = orderstr_dict.get(order_num, ' '+str(order_num+1)+'th ')
    return order_str

def get_mem_size(format=True):
    import torch

    mems_per_gpu = {'gpu'+str(i): torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())}
    mems_map = {
        key: reduce_func(mems_per_gpu.values())
        for key, reduce_func in {'max':max, 'min':min, 'sum':sum}.items()
    }
    mems_map.update(mems_per_gpu)

    if not format: return mems_map

    for k, v in mems_map.items():
        gb_size = v // (1024 ** 3)
        mb_size = (v % (1024 ** 3)) // (1024 ** 2)
        kb_size = (v % (1024 ** 2)) // 1024
        mems_map[k] = f"{gb_size}GB {mb_size}MB {kb_size}KB"

    return mems_map

def info_dict(d, t=1, precision=2) -> str:
    s = "{\n"
    for k, v in d.items():
        s += "\t"*t + str(k)
        s += " : "
        if isinstance(v, dict):
            vd = info_dict(v, t+1)
            s += vd
        else:
            if isinstance(v, float):
                if len(str(v)) > len("0.001"): s += f"{v:.{precision}e}"
                else: s += str(v)
            else: s += str(v)
                    
        s += "\n"
    s +=  "\t"*(t-1) + "}"

    return s

def find_context_len(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path): return "UNKNOWN"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    context_len_keys = [
        "max_position_embeddings",
        'seq_length',
        "seq_len",
        "n_positions",
    ]
    for key in context_len_keys:
        if key in config: return config[key]
    else: return "UNKNOWN"

def retrieve_highest(format_str, precision=0):
    """retrieve only the highest non-zero number in format_str, precision = 0, 1, 2, ...
    e.g. 
        '0GB 114MB 5KB' => '114MB' (precision = 0)
        '2GB 1114MB' => '2.9GB' (precision = 1)
    """
    import re
    
    elems = format_str.split()
    for idx, elem in enumerate(elems):
        match = re.findall(r'\d+', elem)
        if match and int(match[0]) > 0:
            if idx == len(elems)-1 or precision==0: 
                return elem
            elif idx < len(elems)-1:
                next_elem = elems[idx+1]
                next_match = re.findall(r'\d+', next_elem)
                if next_match and int(next_match[0]) > 0:
                    val = float(match[0]) + float(next_match[0]) / 1024
                    unit =  elem[len(match[0]):]
                    return f"{val:.{precision}f}{unit}"
            
    return format_str

def format_context_len(context_len: int) -> str:
    if context_len < 1024: return str(context_len)
    k_size = context_len // 1024
    return f"{k_size}k"

def get_model_name_from_path(model_path):
    return os.path.basename(model_path)

def get_model_base_from_path(model_path, model_root):
    return model_path.split('/')[len(model_root.split('/'))-1]