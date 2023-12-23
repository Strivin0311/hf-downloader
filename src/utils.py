import os

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
    