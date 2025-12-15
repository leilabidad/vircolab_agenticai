from tqdm import tqdm

def progress(iterable, enabled):
    return tqdm(iterable) if enabled else iterable
