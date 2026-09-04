import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader , random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from pathlib import Path
from Dataset_Translate import TranslateDataset , casual_mask 

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

def build_dataset(config) :
    dataset_raws = load_dataset("opus_book" , "en-nl" , split='train')
    source_tokenizer = tokenizer_builder(config , dataset_raws ,'en')
    target_tokenizer = tokenizer_builder(config , dataset_raws ,'nl')
    train_dataset_size = int(0.9 * len(dataset_raws))
    validation_dataset_size = len(dataset_raws) - train_dataset_size
    train_dataset_raw , validation_dataset_raw = random_split(dataset_raws , [train_dataset_size , validation_dataset_size])
    train_dataset = TranslateDataset(train_dataset_raw , source_tokenizer , target_tokenizer , config['seq_len'] , 'en' , 'nl' ) 
    validation_dataset = TranslateDataset(validation_dataset_raw , source_tokenizer , target_tokenizer , config['seq_len'] , 'en' , 'nl' )

    max_len_source = 0
    max_len_target = 0
    for item in dataset_raws :
        source_ids = source_tokenizer.encode(item['translation'][config['src_language']]).ids
        target_ids = target_tokenizer.encode(item['translation'][config['trg_language']]).ids
        max_len_source = max(max_len_source , len(source_ids))
        max_len_target = max(max_len_target , len(target_ids))
    print(f'Max length of source:{max_len_source}')
    print(f'Max length of target:{max_len_target}')
    train_dataloader = DataLoader(train_dataset , batch_size=config['batch_size'] , shuffle=True)
    validation_dataloader = DataLoader(validation_dataset , batch_size=1 , shuffle=True)
    return train_dataloader , validation_dataloader , source_tokenizer , target_tokenizer    