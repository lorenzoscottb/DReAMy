
from datasets import load_dataset
import pandas as pd

DB_dreams_HF = "DReAMy-Library/DreamBank-dreams"

def get_HF_DreamBank(as_dataframe=True):
	
	if as_dataframe:
		data = pd.DataFrame(load_dataset(DB_dreams_HF)["train"])
		
	else:
		data = load_dataset(DB_dreams_HF)
	return data

