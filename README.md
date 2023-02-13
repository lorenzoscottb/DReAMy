

![dreamy_logo](images/dreamy_full_logo.png)

DReAMy is a python library built around pytorch and Hugging Face's 🤗 [`transformers`](https://huggingface.co/docs/transformers/index) to automatically analyse (for now) textual reports of dream. At the moment, annotations are based on the [Hall & Van de Castle](https://link.springer.com/chapter/10.1007/978-1-4899-0298-6_2) emotions framewokr, but we are looking forward to expand our set applications. 

# Installation and usage
DReAMy can be easly installed via pip! If you wish to play/query a set of DReAMy's model, you can do so in the [`dream`](https://huggingface.co/spaces/DReAMy-lib/dream) 🤗 Space.
```
pip install dreamy
```

# Current Feature
## Dataset
You can download and use a collection of (previously scraped) dreams from the DreamBank database (~29 k).
```py
import dreamy

dream_bank = dreamy.get_HF_DreamBank(as_dataframe=True)

n_samples     = 10
dream_sample  = dream_bank.sample(n_samples, random_state=35).reset_index(drop=True)
dream_as_list = dream_sample["dreams"].tolist()

dream_bank.sample(2)
```
|index|dreams|series|description|
|---|---|---|---|
|8726|I was sitting on the porch in the sun\. It was cool and I was thinking how nice it would be if I could take off my shirt and take a sunbath\.|pegasus|Pegasus: a factory worker|
|22410|I couldn't make the elevator work\. Had trouble first to bring it down and then even when I pressed three buttons, it didn't move\. Two men came to help\. Then I am walking along a slippery grass path on the high bank of a stream\. Then I cross a gravelly field thinking it's lucky I have on my old shoes\. I ask the woman with me how much further\. Is it to that new building over there? "Yes\." As we approach I see indications that the field is being prepared for a lawn\.|dorothea|Dorothea: 53 years of dreams|

## Emotion Classification
DReAMy comes equipped with a set of model tuned  to reproduce expert-annotators labels accoding to the [Hall & Van de Castle](https://dreams.ucsc.edu/Coding/) system. These models can perform emotion classification. (a.k.a. sentiment analysis) following 2 main patterns.
### Presence 
Choose between two models:
#### English-only (base)
```py
classification_type = "presence"
model_type          = "base-en"
```
#### Multilingual (large)
```py
classification_type = "presence"
model_type          = "large-multi"
```
and run 
```py 
return_all_scores   = True
device              = "cpu"
max_length          = 512
truncation          = True
device              = "cpu"

model_name, task = dreamy.emotion_classification.model_maps[
    "{}-{}".format(classification_type, model_type)
]

predictions = dreamy.predict_emotions(
    dream_as_list, 
    model_name, 
    task,
    return_all_scores=return_all_scores, 
    max_length=max_length, 
    truncation=truncation, 
    device=device,
)

predictions
```
```
[[{'label': 'AN', 'score': 0.08541450649499893},
  {'label': 'AP', 'score': 0.1043919175863266},
  {'label': 'SD', 'score': 0.029732409864664078},
  {'label': 'CO', 'score': 0.18161173164844513},
  {'label': 'HA', 'score': 0.30588334798812866}],
 [{'label': 'AN', 'score': 0.11174352467060089},
  {'label': 'AP', 'score': 0.17271170020103455},
  {'label': 'SD', 'score': 0.026576947420835495},
  {'label': 'CO', 'score': 0.1214553639292717},
  {'label': 'HA', 'score': 0.22257845103740692}]
```
You can get the HVDC decodings via 
```py
dreamy.Coding_emotions
```
```
{'AN': 'anger',
 'AP': 'apprehension',
 'SD': 'sadness',
 'CO': 'confusion',
 'HA': 'happiness'}
```
#### Generation
Under this variance, a T5 model is trained with the same data to generate the emotion-based reports, with two extra feature. First, the emotion are "numbered". This refers to the fact that if the same emotion was found more than once in the same report, the model should be able to identify so. Second, the model is also trained to recognise *to which character* the emotion are associated with. See the examples below.
##### English-only, characters + numbered emotions
```py 
classification_type = "generation"

# The remaining arguments are the same
model_type          = "base-en"
device              = "cpu"
max_length          = 512
truncation          = True
device              = "cpu"

model_name, task = dreamy.emotion_classification.model_maps[
    "{}-{}".format(classification_type, model_type)
]

predictions = dreamy.generate_emotions(
    dream_as_list, 
    model_name, 
    task,
    max_length=max_length, 
    truncation=truncation, 
    device=device,
)

predictions
```
```
[{'summary_text': 'the dreamer experienced sadness. the individual male father adult experienced apprehension . the group indefinite known adult experienced sadness .'},
 {'summary_text': 'The dreamer experienced happiness and apprehension and happiness. The individual female known adult experienced sadness. the dreamer and the individual female uncertian adult experienced happiness'}]
```
## NER

## Encoding, reduction and visualisation
You can also use DReAMy to easily extract, reduce, cluster, and visualise encodings (i.e., vector embeddings) of dream reports, with few and simple lines of codee. At the moment, you can chose betweem four model, that are combination of small/large Engish-ony/multilingual models.

```py
model_size = "small"   # or large
model_lang = "english" # or multi, for multilingual
device     = "cpu"     # if available, select "cuda" to use GPUs

report_encodings = dreamy.get_encodings(
    dream_list, 
    model_size=model_size,
    language=model_lang, 
    device=device,
)

X, Y = dreamy.reduce_space(report_encodings, method="pca") # choose between pca/t-sne

# Update your original dataframe with cohordinates and plot
dream_sample["DR_X"], dream_sample["DR_Y"] = X, Y
```
Then use your favourite visualisation library to explore the results.
```py

import seaborn as sns

sns.set_context("talk")
sns.set_style("whitegrid")

g = sns.scatterplot(
    data=dream_sample, 
    x="DR_X", 
    y="DR_Y", 
    hue="series",
    palette="Set2"
)
g.legend(loc='center left', title="DreamBank Series", bbox_to_anchor=(1, 0.5))

```
![alt text](https://github.com/lorenzoscottb/DReAMy/blob/main/images/dreamy_example.png)

## In-Progress Development
### Topic-Modelling

## Planned Development
### Audio-to-Text pipeline
### EEG interface

## Contribute

## Cite 
