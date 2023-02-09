
author_ = "lb540"

import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers 

def set_seed(seed: int, set_random=True):
    """Helper function for reproducible behavior to set the seed in ``random``, 
        ``numpy``, ``torch`` and/or ``tf`` (if installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
    
    if set_random:
        random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)
        
def decode_clean(x, tokenizer):
    s = tokenizer.decode(x).replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "")
    return s


def preprocess_function(examples, tokenizer, ssource_clm, target_clm, prefix="", max_source_length=512, max_target_length=128):
    inputs = [prefix + doc for doc in examples[source_clm]]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[target_clm], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


    

    