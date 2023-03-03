
"""
Names/acess codes to the HF models
"""

# Hugging Face models name
t_5_generation_NER = "DReAMy-lib/t5-base-DreamBank-Generation-NER-Char"

# Classification_type+model_type to model + task (for pipleine)
_ner_model_maps = {
	"full-base-en": [t_5_generation_NER, "summarization"],
}

# Model description to HF model name
_ner_models_dict = {
	"Base En-only, generats full cahractes descriptions (T5-base)": t_5_generation_NER,
}

# Description to classification_type, model_type (for pipleine)
_ner_models_dict = {
	"Base En-only, generats full cahractes descriptions (T5-base)": ["full", "base-en"],
}

# Print functions
def get_ner_model_specifications():

	return _ner_models_dict

def get_ner_HF_names():

	return _ner_models_HF_names
