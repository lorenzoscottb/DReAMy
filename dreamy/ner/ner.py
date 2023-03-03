
from ..util.util import decode_clean_T5
from .models_names import _ner_model_maps

def get_CHAR(data, classification_type, model_type, max_length=512, max_new_token=128, truncation=True, device="cpu"):

    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
    
    local_model_map  = "{}-{}".format(classification_type, model_type)
    model_name, task = _ner_model_maps[local_model_map]
 
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if local_model_map not in _ner_model_maps.keys():
        print("Sorry, no model for such method")
        return None

    else:
        inputs = tokenizer(data, max_length=max_length, padding=True, truncation=truncation, return_tensors="pt")
        generation_output = model.generate(**inputs, max_new_tokens=max_new_token)
        predictions = [decode_clean_T5(inpt, tokenizer) for inpt in generation_output]

    return predictions