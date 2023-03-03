
from .models_names import _emotion_model_maps
from ._presence_text_classification import get_predictions_via_pipeline
from ._test_best_model import test_best_model


def predict_emotions(data, classification_type, model_type, return_type="distribution", max_length=512, truncation=True, device="cpu"):

    local_model_map  = "{}-{}".format(classification_type, model_type)
    model_name, task = _emotion_model_maps[local_model_map]

    if local_model_map == "presence-large-en":
        predictions = test_best_model(data, device, max_length)
        
    elif local_model_map not in _emotion_model_maps.keys():
        print("Sorry, no model for such method")
        return None

    else:
        predictions = get_predictions_via_pipeline(data, model_name, task, return_type, max_length, truncation, device)

    return predictions
