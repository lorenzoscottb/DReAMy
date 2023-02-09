

def get_predictions_via_pipeline(dreams_list, model_name, task, return_all_scores, max_length, truncation, device):
	
	"""
	data: list of strings
	model_name: string
	return_all_scores: bool
	max_length: int
	truncation: bool
	"""
	
	from transformers import pipeline

	pipe = pipeline(
	    task,
	    model=model_name,
	    tokenizer=model_name, 
	    return_all_scores=return_all_scores,
	    max_length=max_length, 
	    truncation=truncation,
	    device=device,
	)

	predictions = pipe(dreams_list)

	return predictions
