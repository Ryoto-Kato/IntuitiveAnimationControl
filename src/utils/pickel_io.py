import pickle
import os

def dump_pckl(data, save_root, pickel_fname):
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)
    
    pickel_file_path = os.path.join(save_root, pickel_fname)

    if not os.path.exists(pickel_file_path):
        f = open(pickel_file_path, 'x')
        f.close()

    with open(pickel_file_path, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def load_from_memory(path_to_memory, pickle_fname):
    """
    Returns @dataclass
    """
    
    path_to_pickel = os.path.join(path_to_memory, pickle_fname)

    
    with open(path_to_pickel, 'rb') as f:
        data = pickle.load(f)
    f.close()
    
    return data