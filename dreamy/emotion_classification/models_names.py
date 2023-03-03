
"""
Names/acess codes to the HF models
"""

# Names of the HUgging face models
best_paper_model          = "DReAMy-lib/DB-custom-architecture"
xlm_roberta_emotion_class = "DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence"
bert_base_emotion_class   = "DReAMy-lib/bert-base-cased-DreamBank-emotion-presence"
t5_genration_char         = "DReAMy-lib/t5-base-DreamBank-Generation-Emot-Char"
t5_genration_number       = "DReAMy-lib/t5-base-DreamBank-Generation-Emot-EmotNn"

# Classification_type+model_type to model + task (for pipleine)
_emotion_model_maps = {
	"presence-large-en": [best_paper_model, "none"],
	"presence-base-en": [bert_base_emotion_class, "text-classification"],
	"presence-large-multi": [xlm_roberta_emotion_class, "text-classification"],
	"generation-char-en": [t5_genration_char, "summarization"],
	"generation-nmbr-en": [t5_genration_number, "summarization"],
}

# Model description to HF model name
_emotion_models_HF_names = {
	"Custom architecture, best for multi-labell classification. (Large) En-only": best_paper_model,
	"Large Multilingual model (XLM-R)": xlm_roberta_emotion_class,
	"English-Only base-model (BERT-base)": bert_base_emotion_class,
	"Base En-only text-generation: characters + emotions (T5-base)": t5_genration_char,
	"Base En-only text-generation: numberd emotions (T5-base)": t5_genration_char,

}

# Description to classification_type, model_type (for pipleine)
_emotion_models_dict = {
	"Custom architecture, best for multi-labell classification. (Large) En-only": ["presence", "large-en"],
	"Large Multilingual model (XLM-R)": ["presence", "large-multi"],
	"English-Only base-model (BERT-base)": ["presence", "base-en"],
	"Base En-only text-generation: Characters + emotions (T5-base)": ["generation", "char-en"],
	"Base En-only text-generation: numbered emotions (T5-base)": ["generation", "nmbr-en"],
}

# Print functions
def get_emotions_model_specifications():

	return _emotion_models_dict

def get_emotions_HF_names():

	return _emotion_models_HF_names
