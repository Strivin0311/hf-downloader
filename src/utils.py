import os
from typing import List, Dict


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
    """string format should be: '{} B {} M' """
    num_params = sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_learnable))
    if not format: return num_params
    
    b_size = num_params // (1000 ** 3)
    m_size = (num_params % (1000 ** 3)) // (1000 ** 2)
    return f"{b_size} B {m_size} M" 

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