import torch
import torch.nn as nn
from torch.utils.data import Dataset
def causual_mask(size):
    mask = torch.triu(torch.ones((1 , size , size)) , diagonal=1).type(torch.int)
    return mask == 0

class TranslateDataset(Dataset):
    def __init__(self , dataset , tokenizer_src , tokenizer_trg , seq_len , src_language = 'en' , trg_language = 'nl'):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.src_language = src_language
        self.trg_language = trg_language
        self.seq_len = seq_len
        self.special_token_sos = torch.tensor([tokenizer_src.token_to_id('[SOS]')] , dtype=torch.int64)
        self.special_token_eos = torch.tensor([tokenizer_src.token_to_id('[EOS]')] , dtype=torch.int64)
        self.special_token_pad = torch.tensor([tokenizer_src.token_to_id('[PAD]')] , dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        source_target = self.dataset[index]
        source_text = source_target['translation'][self.src_language]
        target_text = source_target['translation'][self.trg_language]
        encoder_input_tokens = self.tokenizer_src.encode(source_text).ids
        decoder_input_tokens = self.tokenizer_trg.encode(target_text).ids # changed source_text to target_text
        padding_number_encoder = self.seq_len - len(encoder_input_tokens) - 2
        padding_number_decoder = self.seq_len - len(decoder_input_tokens) - 1

        # This check is now redundant if filtering is done before creating the dataset
        # if padding_number_decoder < 0 or padding_number_encoder < 0 :
        #     raise ValueError ('sequence is long')

        # Make sure that the sequence length is not negative after padding, if it is just set it to 0.
        padding_number_encoder = max(0, padding_number_encoder)
        padding_number_decoder = max(0, padding_number_decoder)

        encoder_input = torch.cat(
            [
            self.special_token_sos , torch.tensor(encoder_input_tokens , dtype=torch.int64) ,
            self.special_token_eos , torch.tensor([self.special_token_pad] * padding_number_encoder , dtype=torch.int64),
            ],
            dim=0 ,
        )
        decoder_input = torch.cat(
            [
                self.special_token_sos ,
                torch.tensor(decoder_input_tokens , dtype=torch.int64),
                torch.tensor([self.special_token_pad] * padding_number_decoder , dtype=torch.int64),
            ],
            dim=0,
        )
        target = torch.cat(
        [
        torch.tensor(decoder_input_tokens , dtype=torch.int64),
        self.special_token_eos , # changed special_token_sos to special_token_eos
        torch.tensor([self.special_token_pad] * padding_number_decoder , dtype=torch.int64),
        ],
        dim=0,
        )

        # assert encoder_input.size(0) == self.seq_len
        # assert decoder_input.size(0) == self.seq_len
        # assert target.size(0) == self.seq_len

        return{
            "encoder_input" : encoder_input ,
            "decoder_input" : decoder_input ,
            "encoder_mask" : (encoder_input != self.special_token_pad).unsqueeze(0).unsqueeze(0).int() ,
            "decoder_mask" : (decoder_input != self.special_token_pad).unsqueeze(0).int() & causual_mask(decoder_input.size(0)) ,
            "label" : target ,
            "source_text" : source_text ,
            "target_text" : target_text
        }