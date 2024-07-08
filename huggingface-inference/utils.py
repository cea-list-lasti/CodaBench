import argparse
import json


def str2bool(v):
    """ Enable boolean in argparse  
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def  save_to_jsonl(file_path, data):
    """ Save a list of dict to json
        Args:
            file_path (_type_): _description_
            data (_type_): _description_
    """
    with open(file_path, 'w', encoding='UTF-8') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')


def read_jsonl(file_path):
    """ Read a jsonl file
        Args: file_path (str): jsonl file path
        Returns: [dict]: data contained in json as list of dicts
    """
    data = []
    with open(file_path, 'r', encoding='UTF-8') as file:
        lines = file.read().splitlines()
        for line in lines:
            item = json.loads(line.strip())
            data.append(item)
    return data

def batch(iterable, n=1):
    """ Batch over an iterable
        Args:
            iterable (iterable): any iterable
            n (int, optional): batch size. Defaults to 1.

        Yields:
            iterable: batches
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]