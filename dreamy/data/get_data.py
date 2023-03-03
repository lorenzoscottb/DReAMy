
from datasets import load_dataset
import pandas as pd

DB_dreams_HF_full = "DReAMy-lib/DreamBank-dreams"
DB_dreams_HF_en   = "DReAMy-lib/DreamBank-dreams-en"

datasets_dict ={
	"english": DB_dreams_HF_en, 
	"multi": DB_dreams_HF_full

}

def get_HF_DreamBank(language="english", as_dataframe=True):
	
	DB_dreams_name = datasets_dict[language]

	if as_dataframe:
		data = pd.DataFrame(load_dataset(DB_dreams_name)["train"])
		
	else:
		data = load_dataset(DB_dreams_name)
	return data
