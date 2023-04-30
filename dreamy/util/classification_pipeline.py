
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

def get_predictions_via_classification_pipeline(HF_dataset, model_name, return_type="distribution", max_length=512, truncation=True, padding="max_length", device=0, batch_size=16):
	
	"""
	data: list of strings
	model_name: string
	return_all_scores: str: "ditribution", for all; "present" for above threshold
	max_length: int
	truncation: bool
	"""
	
	from transformers import pipeline

	pipe = pipeline(
	    "text-classification",
	    model=model_name,
	    tokenizer=model_name, 
	    top_k=None,
	    device=device,
	    max_length=max_length, 
	    truncation=truncation,
	    padding=padding
	)

	predictions = []
	for out in tqdm(pipe(KeyDataset(HF_dataset, "dreams"), batch_size=batch_size)):
	    predictions.append(out)

	if return_type == "present":
	  predictions = [
	    [
	    emotion_score_dict["label"] for emotion_score_dict in emotion_score_list 
	    if emotion_score_dict["score"] > .5
	    ]
	    for emotion_score_list in predictions
	  ]

	return predictions