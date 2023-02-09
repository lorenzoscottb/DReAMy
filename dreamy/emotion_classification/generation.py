
from ..util.util import decode_clean
from ..data import Dataset
from ..util.util import preprocess_function

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


def generate_emotions(data, model_name, task, max_length=512, truncation=True, device="cpu"):

    from transformers import pipeline

    pipe = pipeline(
        task,
        model=model_name,
        tokenizer=model_name, 
        max_length=max_length, 
        truncation=truncation,
        device=device,
    )

    predictions = pipe(data)

    return predictions


