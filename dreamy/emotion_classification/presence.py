
from ._presence_text_classification import get_predictions_via_pipeline
from ._test_best_model import test_best_model


def predict_emotions(data, model_name, task, return_all_scores=True, max_length=512, truncation=True, device="cpu"):

    if "DB-custum" in model_name:
        predictions =test_best_model(data_list, device, max_length)
        
    elif "presence":
        predictions = get_predictions_via_pipeline(data, model_name, task, return_all_scores, max_length, truncation, device)

    else:
        print("Sorry, no model for such method")
        return None

    return predictions

