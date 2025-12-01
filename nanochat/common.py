import os

def get_base_dir():
    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, ".cache", "nanochat")