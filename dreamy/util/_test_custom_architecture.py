

from .architecture import BERTClass
from ..data.custom_datasets import Train_Dataset
from .hvdc_decodings import Coding_emotions
from .train_util import validation
from .util import decode_clean

import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader


def test_best_model(data_list, device, max_length):

    # Setup model and tokenizer
    path_to_downloaded_model = hf_hub_download(
        repo_id="DReAMy-lib/DB-custom-architecture", 
        filename="pytorch_model.bin"
    )
    
    model_name = "bert-large-cased"
    tokenizer  = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

    model = BERTClass(
        model_name=model_name, 
        n_classes=5, 
        freeze_BERT=False,
    )
    model.load_state_dict(torch.load(path_to_downloaded_model, map_location=torch.device('cpu')))

    if device == "cuda":
        model.to(device)

    # Setup data
    emotions_list = list(Coding_emotions.keys())

    test_sentences_target = len(data_list)*[[0, 0, 0, 0, 0]]
    test_sentences_df     = pd.DataFrame.from_dict(
                    {"report":data_list,
                    "Report_as_Multilabel":test_sentences_target}
    )

    test_params = {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 0
    }
    
    testing_set     = Train_Dataset(test_sentences_df, tokenizer, max_length=max_length)
    testing_loader  = DataLoader(testing_set, **test_params)


    # Get predictions
    outputs, targets, ids = validation(model, testing_loader, device=device, return_inputs=True)
    corr_outputs_df       = pd.DataFrame(np.array(outputs), columns=emotions_list)

    return corr_outputs_df
