

def get_predictions_via_pipeline(dreams_list, model_name, task, return_type, max_length, truncation, device):
	
	"""
	data: list of strings
	model_name: string
	return_all_scores: str: "ditribution", for all; "present" for above threshold
	max_length: int
	truncation: bool
	"""
	
	from transformers import pipeline

	pipe = pipeline(
	    task,
	    model=model_name,
	    tokenizer=model_name, 
	    top_k=None,
	    max_length=max_length, 
	    truncation=truncation,
	    device=device,
	)

	predictions = pipe(dreams_list)

	if return_type == "present":
		predictions = [
			[
			emotion_score_dict["label"] for emotion_score_dict in emotion_score_list 
			if emotion_score_dict["score"] > .5
			]
			for emotion_score_list in predictions
		]

	return predictions
