

![dreamy_logo](images/dreamy_full_logo.png)

DReAMy is a python library built around pytorch and Hugging Face's 🤗 [`transformers`](https://huggingface.co/docs/transformers/index) to automatically analyse (for now only) textual dream reports. At the moment, annotations are based on the [Hall & Van de Castle](https://link.springer.com/chapter/10.1007/978-1-4899-0298-6_2) emotions framework, but we are looking forward to expand our set applications. The theoretical backbone of DReAMy and its model is based on a fuiifull collaboration between NLP and sleep research. More detailed results can be found [here](https://github.com/lorenzoscottb/Dream_Reports_Annotation).

# Installation and usage
DReAMy can be easily installed via pip! If you wish to play/query a set of DReAMy's model, you can do so in the [`dream`](https://huggingface.co/spaces/DReAMy-lib/dream) 🤗 Space.
```
pip install dreamy
```

# Current Features
At the moment, DReAMy has four main features: 
- Datasets, allowing to download and use two dream-report datasets from DreamBank.
- Emotion Classification, allowing to easily classify lists of reports for HVDC emotions.
- NER (or character annotation), that allow to extract relevant characters appearing in a given report. 
- Encodings: that easily collects and explores embeddings of textual reports.

Use example can be found in the code below, and in the tutorial folder. 

## Dataset
DReAMy has direct access to two datasets. A smaller English-only (~ 20k), with more descriptive variables (such as gender and year), and a larger and multilingual one (En/De, ~ 30 k). You can easily choose between the two of them with the simple code below.

```py
import dreamy

language   = "english" # choose between english/multi
dream_bank = dreamy.get_HF_DreamBank(as_dataframe=True, language=language)

n_samples     = 10
dream_sample  = dream_bank.sample(n_samples, random_state=35).reset_index(drop=True)
dream_as_list = dream_sample["dreams"].tolist()

dream_bank.sample(2)
```
|index|series|description|dreams|gender|year|
|---|---|---|---|---|---|
|5875|blind-f|Blind dreamers \(F\)|I'm at work in the office of a rehab teacher named D, a transistor radio is on, [...]\.|female|mid-1990s|
|12888|emma|Emma: 48 years of dreams|I go to Pedro's house, he is fixing his bike\. I think I will take my bike out too, but [...]\.|female|1949-1997|

## Emotion Classification
DReAMy comes equipped with a set of model tuned  to reproduce expert-annotators labels accoding to the [Hall & Van de Castle](https://dreams.ucsc.edu/Coding/) system. These models can perform emotion classification. (a.k.a. sentiment analysis) following 2 main patterns.
### Presence 
Two model are currently available to detect the presence of difference emotions: `base-en` and `large-multi`, easily querible with the short code below.
```py 
classification_type = "presence"
model_type          = "base-en"
return_type         = "distribution" # set "present" for above-threshold only
device              = "cpu"

predictions = dreamy.predict_emotions(
    dream_as_list, 
    classification_type, 
    model_type,
    return_type=return_type, 
    device=device,
)

predictions
```
```
[[{'label': 'CO', 'score': 0.7488341331481934},
  {'label': 'HA', 'score': 0.09567967802286148},
  {'label': 'AN', 'score': 0.03418444097042084},
  {'label': 'AP', 'score': 0.019197145476937294},
  {'label': 'SD', 'score': 0.012466167099773884}],
 [{'label': 'HA', 'score': 0.9818947911262512},
  {'label': 'SD', 'score': 0.03642113506793976},
  {'label': 'AP', 'score': 0.03470420092344284},
  {'label': 'CO', 'score': 0.024184534326195717},
  {'label': 'AN', 'score': 0.023663492873311043}]
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
Under this variance, a T5 model is trained with the same data to generate the emotion-based reports, with two extra feature. First, the emotion are "numbered". This refers to the fact that if the same emotion was found more than once in the same report, the model should be able to identify so. Second, the model is also trained to recognize *to which character* the emotion are associated with. See the examples below.
##### English-only, characters + numbered emotions
```py 
classification_type = "generation"
model_type          = "char-en"
device              = "cpu"

predictions = dreamy.generate_emotions(
    dream_as_list, 
    classification_type, 
    model_type,
    device=device,
)

predictions
```
```
['The dreamer experienced apprehension.',
 'The group joint stranger adult experienced happiness. The dreamer experienced anger.',]
```
## NER
An important aspect of each dream report is the character that appear in it. In this notebook, we will see how to use `dreamy` to extract character appearing in each report. As always, character are defined with respect to the Hall & Van de Castle system. CHAR are in this case spelled out, and do not/should not include the dreamer themself. Please note that CHAR data used in training is not linked to any specii feature. In other words, prediction should not be interpreted in any other way other than their presence. 
```py 
classification_type = "full"
model_type          = "base-en"
device              = "cpu"

predictions = dreamy.get_CHAR(
    dream_as_list, 
    classification_type, 
    model_type,
    device=device,
    max_new_token=60,
)
predictions
```
```
['individual female known adult; group female uncertian adult; individual indefinite uncertian adult;',
 'individual female known adult;']
```

## Encoding, reduction and visualisation
You can also use DReAMy to easily extract, reduce, cluster, and visualize encodings (i.e., vector embeddings) of dream reports, with few and simple lines of codee. At the moment, you can chose betweem four model, that are combination of small/large English-ony/multilingual models.

```py
import dreamy

# get some data
n_samples  = 10
language   = "english" # choose between english/multi

dream_bank = dreamy.get_HF_DreamBank(as_dataframe=True, language=language)
dream_bank = dream_bank.sample(n_samples).reset_index(drop=True)

dream_as_list = dream_sample["dreams"].tolist()

# set up model and  get encodings
model_size = "small"   # or large
model_lang = "english" # or multi, for multilingual
device     = "cpu"     # se to "cuda" for GPUs

report_encodings = dreamy.get_encodings(
    dream_as_list, 
    model_size=model_size,
    language=model_lang, 
    device=device,
)

# reduce space
# you can choose between pca/t-sne
X, Y = dreamy.reduce_space(report_encodings, method="pca") 

# Update your original dataframe with cohordinates and plot
dream_bank["DR_X"], dream_bank["DR_Y"] = X, Y
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
Check the starting tutorial for more, like unuspervised K-Mean clustering.
![alt text](https://github.com/lorenzoscottb/DReAMy/blob/main/images/dreamy_example.png)

## In-Progress Development
### Topic-Modelling

## Planned Development
### Audio-to-Text pipeline
### EEG interface

## Contribute
If you wish to contribute, collaborate, or just ask question, feel free to contact [Lorenzo](https://lorenzoscottb.github.io/), or use the issue section.

## Cite 
If you use DReAMy, please cite the pre-print
```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.14828,
  doi = {10.48550/ARXIV.2302.14828},
  url = {https://arxiv.org/abs/2302.14828},
  author = {Bertolini, Lorenzo and Elce, Valentina and Michalak, Adriana and Bernardi, Giulio and Weeds, Julie},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Automatic Scoring of Dream Reports' Emotional Content with Large Language Models},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}

```
