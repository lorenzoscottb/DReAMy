
from datasets import load_dataset
import pandas as pd

DB_dreams_HF_full = "DReAMy-lib/DreamBank-dreams"
DB_dreams_HF_en   = "DReAMy-lib/DreamBank-dreams-en"

datasets_dict = {
	"base": DB_dreams_HF_en, 
	"large":   DB_dreams_HF_full
}

def get_HF_DreamBank(database="base", as_dataframe=True):
	
	DB_dreams_name = datasets_dict[database]
	data 		   = load_dataset(DB_dreams_name)
	
	if as_dataframe:
		data = pd.DataFrame(data["train"])
		
	return data
