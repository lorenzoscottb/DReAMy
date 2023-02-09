
"""
Names/acess codes to the HF models
"""

t_5_generation_NER = "DReAMy-lib/t5-base-DreamBank-Generation-NER-Char"


ner_models_dict = {
	"NER via text generation of full cahractes descriptions": t_5_generation_NER,
}

def get_NER_names():
	print(ner_models_dict)


ner_model_maps = {
	"full-base-en": [t_5_generation_NER, "summarization"],
}

def get_NER_maps():
	print(ner_model_maps)
