
"""
Names/acess codes to the HF models
"""

best_paper_model          = "DReAMy-lib/DB-custom-architecture"
xlm_roberta_emotion_class = "DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence"
bert_base_emotion_class   = "DReAMy-lib/bert-base-cased-DreamBank-emotion-presence"
t5_genration_char         = "DReAMy-lib/t5-base-DreamBank-Generation-Emot-Char"


models_dict = {
	"Custom architecture, best for multi-labell classification": best_paper_model,
	"Large Multilingual model (XLM-R)": xlm_roberta_emotion_class,
	" (base) English-Only model (BERT-base)": bert_base_emotion_class,
	"Text-Generation (T5), Characters + numberd emotions": t5_genration_char
}

def get_names():
	print(models_dict)


model_maps = {
	"presence-large-en": [best_paper_model, "none"],
	"presence-base-en": [bert_base_emotion_class, "text-classification"],
	"presence-large-multi": [xlm_roberta_emotion_class, "text-classification"],
	"generation-base-en": [t5_genration_char, "summarization"],
}

def get_maps():
	print(model_maps)
