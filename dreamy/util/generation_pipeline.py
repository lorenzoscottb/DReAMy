from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

def get_predictions_via_generation_pipeline(HF_dataset, model_name, min_length=5, max_length=128, device=0, batch_size=16):
	
	"""
	data: list of strings
	model_name: string
	return_all_scores: str: "ditribution", for all; "present" for above threshold
	max_length: int
	truncation: bool
	"""
	
	from transformers import pipeline

	
	pipe = pipeline(
	    task="text2text-generation",
	    model=model_name, 
	    tokenizer=model_name,
	    device=device,
	)

	predictions = []
	for out in tqdm(pipe(KeyDataset(HF_dataset, "dreams"), min_length=min_length, max_length=max_length, batch_size=batch_size)):
	    predictions.append(out[0]["generated_text"])

	return predictions