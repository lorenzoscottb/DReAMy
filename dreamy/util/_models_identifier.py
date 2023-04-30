
# Names of the HUgging face models
## SA
best_paper_model          = "DReAMy-lib/DB-custom-architecture"
xlm_roberta_emotion_class = "DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence"
bert_base_emotion_class   = "DReAMy-lib/bert-base-cased-DreamBank-emotion-presence"
t5_genration_char         = "DReAMy-lib/t5-base-DreamBank-Generation-Emot-Char"
t5_genration_number       = "DReAMy-lib/t5-base-DreamBank-Generation-Emot-EmotNn"

## NER
t_5_generation_NER = "DReAMy-lib/t5-base-DreamBank-Generation-NER-Char"


## RE
t_5_generation_RE = "DReAMy-lib/t5-base-DreamBank-Generation-Act-Char"



model_ids = {
	"NER, EN-only, base, generation of list of (interpretable characters). T5)" : t_5_generation_NER,
	"SA, En-only, large, multi-label classification. Custom architecture from Bertolini et al., 2023" : best_paper_model,
	"SA, Multilingual (94), large, multi-label classification. XLM-RoBERTa" : xlm_roberta_emotion_class,
	"SA, En-only, base, multi-label classification. BERT-base-cased" : bert_base_emotion_class,
	"SA, En-only, base, generation of emotion and character experiencing them. T5" : t5_genration_char,
	"SA, En-only, base, generation of emotion with ammount of presence. T5" : t5_genration_number,
	"RE, EN-only, base, generation of  (initialiser : activity type : receiver) list. T5)" : t_5_generation_RE, 
}

_default_models = {
	"NER":t_5_generation_NER,
	"SA":xlm_roberta_emotion_class,
	"RE":t_5_generation_RE,
}

_models_to_pipeline ={
	"DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence" : "text-classification",
	"DReAMy-lib/bert-base-cased-DreamBank-emotion-presence" : "text-classification",
	"DReAMy-lib/t5-base-DreamBank-Generation-Emot-Char" : "text2text-generation",
	"DReAMy-lib/t5-base-DreamBank-Generation-Emot-EmotNn" : "text2text-generation",
	"DReAMy-lib/t5-base-DreamBank-Generation-NER-Char" : "text2text-generation",
	"DReAMy-lib/t5-base-DreamBank-Generation-Act-Char" : "text2text-generation",
}

def show_models():
	return model_ids

def show_default_models():
	return _default_models