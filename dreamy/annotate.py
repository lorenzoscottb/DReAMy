
from .util._models_identifier import * 
from .util._models_identifier import _default_models, _models_to_pipeline
from .util._test_custom_architecture import *
from .util.classification_pipeline import *
from .util.generation_pipeline import *

def annotate_reports(list_of_reports, task="SA", model_name="default", device=0, batch_size=16, return_type="distribution", max_seq_length=512, truncation=True, padding="max_length", min_out_length=5, max_out_length=128): 

	import datasets
	import pandas as pd

	df = pd.DataFrame(
	    list_of_reports,
	    columns=['dreams']
	)

	HF_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=df))

	if model_name == "default":
		model_name = _default_models[task]

	if "custom" in model_name:

		predictions = test_best_model(list_of_reports, device, max_seq_length)

		return predictions

	elif task == "SA":		
		predictions = get_predictions_via_classification_pipeline(
			HF_dataset, 
			model_name, 
			return_type=return_type, 
			max_length=max_seq_length, 
			truncation=truncation, 
			padding=padding, 
			device=device, 
			batch_size=batch_size
		)

		return predictions

	elif task in ["NER", "RE"]:
		predictions = get_predictions_via_generation_pipeline(
			HF_dataset, 
			model_name, 
			min_length=min_out_length, 
			max_length=max_out_length, 
			device=device, 
			batch_size=batch_size
		)

		return predictions
		
	else:
		print("Error, no such task: {}".format(task))