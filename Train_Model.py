
import torch
import torch.nn as nn
import tqdm
import warnings
from torch.utils.data import Dataset , DataLoader , random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from pathlib import Path
from Transformer_model import build_model
from config import latest_weight_path , get_config , get_weights_path
from Dataset_Translate import TranslateDataset , causual_mask

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torchmetrics
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.generation(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_dataloader, tokenizer_source, tokenizer_target, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    print_msg(f"\n--- Running Validation at Global Step {global_step} ---")
    print_msg(f"Validation Dataloader size: {len(validation_dataloader)}")

    if len(validation_dataloader) == 0:
        print_msg("No validation examples available to process.")
        return

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_dataloader:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_source, tokenizer_target, max_len, device)

            source_text = batch["source_text"][0] # changed src_text to source_text
            target_text = batch["target_text"][0] # changed tgt_text to target_text
            model_out_text = tokenizer_target.decode(model_out.detach().cpu().numpy(), skip_special_tokens=False)

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg(f"Processed {num_examples} validation examples.")
                break

        if count < num_examples:
            print_msg(f"Only {count} validation examples were available out of {num_examples} requested.")

    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

    print_msg("--- Validation Finished ---\n")

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
    dataset_raws = load_dataset("Helsinki-NLP/opus_books" , "en-nl" , split='train')

    # Filter out samples that are too long for the defined sequence length
    seq_len = config['seq_len']
    filtered_dataset_raws = []
    source_tokenizer_tmp = tokenizer_builder(config, dataset_raws, config['lang_src'])
    target_tokenizer_tmp = tokenizer_builder(config, dataset_raws, config['lang_tgt'])
    for item in dataset_raws:
        source_text = item['translation'][config['lang_src']]
        target_text = item['translation'][config['lang_tgt']]
        if len(source_tokenizer_tmp.encode(source_text).ids) < seq_len - 2 and \
           len(target_tokenizer_tmp.encode(target_text).ids) < seq_len - 1:
            filtered_dataset_raws.append(item)

    # Use the filtered dataset for further processing
    from datasets import Dataset as HFDataset # Alias to avoid confusion with torch.utils.data.Dataset
    dataset_raws = HFDataset.from_list(filtered_dataset_raws)
 

    source_tokenizer = tokenizer_builder(config , dataset_raws ,'en')
    target_tokenizer = tokenizer_builder(config , dataset_raws ,'nl')
    train_dataset_size = int(0.9 * len(dataset_raws))
    validation_dataset_size = len(dataset_raws) - train_dataset_size

    print(f"Total dataset size after filtering: {len(dataset_raws)}")
    print(f"Training dataset size: {train_dataset_size}")
    print(f"Validation dataset size: {validation_dataset_size}")

    train_dataset_raw , validation_dataset_raw = random_split(dataset_raws , [train_dataset_size , validation_dataset_size])
    train_dataset = TranslateDataset(train_dataset_raw , source_tokenizer , target_tokenizer , config['seq_len'] , 'en' , 'nl' )
    validation_dataset = TranslateDataset(validation_dataset_raw , source_tokenizer , target_tokenizer , config['seq_len'] , 'en' , 'nl' )

    max_len_source = 0
    max_len_target = 0
    for item in dataset_raws : # Iterate over the filtered dataset_raws
        source_ids = source_tokenizer.encode(item['translation'][config['lang_src']]).ids
        target_ids = target_tokenizer.encode(item['translation'][config['lang_tgt']]).ids
        max_len_source = max(max_len_source , len(source_ids))
        max_len_target = max(max_len_target , len(target_ids))
    print(f'Max length of source:{max_len_source}')
    print(f'Max length of target:{max_len_target}')
    train_dataloader = DataLoader(train_dataset , batch_size=config['batch_size'] , shuffle=True)
    validation_dataloader = DataLoader(validation_dataset , batch_size=1 , shuffle=True)
    return train_dataloader , validation_dataloader , source_tokenizer , target_tokenizer

def get_model(config , vocab_source_len , vocab_target_len):
    model = build_model(vocab_source_len , vocab_target_len , config['seq_len'] , config['d_model'])
    return model

def train_transformer(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    train_dataloader , validation_dataloader , source_tokenizer , target_tokenizer = build_dataset(config)
    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weight_path(config) if preload == 'latest' else get_weights_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    loss_fn = nn.CrossEntropyLoss(ignore_index=target_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm.tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.generation(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        run_validation(model, validation_dataloader, source_tokenizer, target_tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_transformer(config)
