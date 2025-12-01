import os

def get_base_dir():
    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, ".cache", "nanochat")

def get_dist_info():
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1