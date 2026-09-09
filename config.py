from pathlib import Path
def get_config():
    return{
    "batch_size" : 3 ,
    "num_epochs" : 5 ,
    "lr" : 1e-4 ,
    "seq_len" : 512 ,
    "d_model" : 512 ,
    "datasource": 'opus_books',
    "lang_src" : "en" ,
    "lang_tgt" : "nl" ,
    "model_folder" : "weights" ,
    "model_basename" : "tmodel_" ,
    "preload" : None ,
    "tokenizer_file" : "tokenizer_{0}.json" , 
    "experiment_name" : "runs"
    }
def get_weights_path(config , epoch):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
def latest_weight_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files= list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0 :
        return None
    weights_files.sort()
    return str(weights_files[-1])
