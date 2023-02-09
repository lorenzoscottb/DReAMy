
"""
Names/acess codes to the HF models
"""

t_5_generation_NER = "DReAMy-lib/t5-base-DreamBank-Generation-NER-Char"


models_dict = {
	"NER via text generation of full cahractes descriptions": t_5_generation_NER,
}

def get_names():
	print(models_dict)
