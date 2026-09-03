import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from pathlib import Path

def sentences(dataset , language):
    for item in dataset :
        yield item['translation'][language]

def tokenizer_builder(congif , dataset , language ):
    tokenizer_path = Path(congif['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path) :
        tokenizer = Tokenizer(WordLevel(unk_token='[UNKNOWN]'))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        train_tokenizer = WordLevelTrainer(special_tokens=["[UNKNOWN]" , "[PAD]" , "[SOS]" , "[EOS]"] , min_frequency=2)
        tokenizer.train_from_iterator(sentences(dataset , language) , trainer=train_tokenizer)
    else :
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer        