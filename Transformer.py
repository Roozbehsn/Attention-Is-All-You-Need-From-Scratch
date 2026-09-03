from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from torch.nn import Embedding
from torch import nn
import torch
import math

input = "I love cats"

class Embed(nn.Module):

    def __init__(self , d_model  , vocab):
        super().__init__()
        self.embeded = nn.Embedding(vocab , d_model)
        self.d_model = d_model
    def forward (self, x):
        return self.embeded(x) *  math.sqrt(self.d_model)  
    
class PositionalEncoding(nn.Module):
    
    def __init__(self , dropout , d_model , seq_len = 500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len , d_model)
        position = torch.arange(0. , seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0 , d_model ,2) * - (math.log(10000.0)/d_model) )
        pe[: , 0::2] = torch.sin(position * div_term)
        pe[: , 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe' , pe)
    def forward(self , x):
        x = x + (self.pe[: , :x.shape[1] , : ]).requires_grad_(False)
        return self.dropout(x)


     

class LayerNormalization(nn.Module) :
    def __init__(self , size , eps=1e-6):
        super().__init__()
        # self.gamma = nn.parameter(torch.ones(1))
        # self.beta = nn.parameter(torch.zeros(1))
        self.gamma = nn.Parameter(size)
        self.beta = nn.Parameter(size)
        self.eps = eps
    def forward(self , x):
        mean = x.mean(-1 , keepdim=True)
        std = x.std(-1 , keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta



class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model , d_ff , dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model , d_ff) # bias automatically is true
        self.w2 = nn.Linear(d_ff , d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self , x):
        return self.w2(self.droput(torch.relu(self.w1(x))))


class LayerNormalization(nn.Module) :
    def __init__(self , size , eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(size)
        self.beta = nn.Parameter(size)
        self.eps = eps
    def forward(self , x):
        mean = x.mean(-1 , keepdim=True)
        std = x.std(-1 , keepdim=True)
        return self.gammas * (x - mean) / (std + self.eps) + self.beta     

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model , d_ff , dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model , d_ff) # bias automatically is true
        self.w2 = nn.Linear(d_ff , d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self , x):
        return self.w2(self.dropout(torch.relu(self.w1(x))))        

class MultiHeadAttentions(nn.Module):

    def __init__(self , d_model , head , dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.dropout = nn.Dropout(p=dropout)
        assert d_model % head == 0
        self.d_k = d_model // head

        self.w_query = nn.Linear(d_model , d_model)
        self.w_key = nn.Linear(d_model , d_model)
        self.w_value = nn.Linear(d_model , d_model)
        self.v_output = nn.Linear(d_model , d_model)

    @staticmethod
    def attention(query , key , value , mask , dropout):
        d_k = query.size(-1)
        attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None :
            attention_score.masked_fill_(mask == 0 , -1e9)
        attention_score = attention_score.softmax(dim = -1)
        if dropout is not None :
            attention_score = dropout(attention_score)
        return (attention_score @ value) , attention_score        
    
    def forward(self , q , k , v , mask):
        query = self.w_query(q)
        key = self.w_key(k)
        value = self.w_value(v)

        query = query.view(query.shape[0] , query.shape[1] , self.head , self.d_k).transpose(1,2)    
        key = key.view(key.shape[0] , key.shape[1] , self.head , self.d_k).transpose(1,2)
        value = value.view(value.shape[0] , value.shape[1] , self.head , self.d_k).transpose(1,2)
        x , self.attention_score = MultiHeadAttentions.attention(query , key , value , mask , self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0] , -1 , self.d_k * self.head)

        return self.v_output(x)

#Also called ResidualConnection
class LayerConnection(nn.Module):
    def __init__(self , dropout):
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self , x , sublayer):
       
       return x + self.dropout(sublayer(self.norm(x)))    
           
class EncoderLayer(nn.Module):

    def __init__(self , attention_block : MultiHeadAttentions , feedforwardblock : FeedForwardNetwork , dropout):
        super().__init__()
        self.attention_block = attention_block
        self.feedforwardblock = feedforwardblock
        self.residualconnection = nn.ModuleList([LayerConnection(dropout) for _ in range(2)])
    def forward(self , x , mask):
        x = self.residualconnection[0](x , lambda x : self.attention_block(x , x , x , mask))
        x= self.residualconnection[1](x , self.feedforwardblock)
        return x

class Encoder(nn.Module):

    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self , x , mask):
        for layer in self.layers :
            x = layer(x , mask)
        return self.norm(x)    
            
class DecoderLayer(nn.Module):
    def __init__(self, attenntion_block : MultiHeadAttentions , cross_attention_block : MultiHeadAttentions , feedforwardblock : FeedForwardNetwork , dropout):
        super().__init__()
        self.attention_block = attenntion_block
        self.cross_attention_block = cross_attention_block
        self.feddforwardblock = feedforwardblock
        self.residualnetwork = nn.ModuleList([LayerConnection(dropout) for _ in range (3)])
    # x : input of Decoder , source_mask : mask applied to the Encoder , target_mask : mask apllied to Decoder    
    def forward(self , x , encoder_output , source_mask , target_mask):
        x = self.residualnetwork[0](x , lambda x : self.attention_block(x , x , x , target_mask))
        x = self.residualconnection[1](x , lambda x : self.cross_attention_block(x , encoder_output , encoder_output , source_mask ))
        x= self.residualconnection[2](x , self.feedforwardblock)
        return x

class Decoder(nn.Module):
    def __init__(self , layer : nn.ModuleList):
        super().__init__()
        self.layers = layer
        self.norm = LayerNormalization()
    def forward(self , x , encoder_output , soucre_mask , target_mask):
        for layer in self.layers :
            x = layer(x , encoder_output , soucre_mask , target_mask)
        return self.norm(x)

class Generation(nn.Module):
    def __init__(self, d_model , vocab_size):
        super().__init__()
        self.gnrt = nn.Linear(d_model , vocab_size)
    def forward(self , x):
        return torch.log_softmax(self.gnrt(x) , dim = -1)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder : Encoder , decoder : Decoder , src_embed : Embed , trg_embed : Embed , src_pos : PositionalEncoding , trg_pos : PositionalEncoding , generator_layer : Generation):
        super().__init__()
        self.encoders = encoder
        self.decoders = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.generator_layer = generator_layer
    def encoder(self , src , src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src , src_mask)
    def decoder(self , trg , trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg , trg_mask)
    def generation(self , x):
        return self.generator_layer(x)

def build_model(src_vocab , trg_vocab , src_seq , trg_seq , d_model = 512 , N = 6 , head = 8 , dropout = 0.1 , d_ff = 2048):
    src_embedding =  Embed(d_model , src_vocab)
    trg_embedding = Embed(d_model ,trg_vocab)
    src_position = PositionalEncoding(d_model , src_seq , dropout)
    trg_position = PositionalEncoding(d_model , trg_seq , dropout )

    encoder_layers = []
    for _ in range(N) : 
        encoder_self_attention = MultiHeadAttentions(d_model , head , dropout)
        encoder_feedforward = FeedForwardNetwork(d_model , d_ff , dropout)
        encoder_layer = EncoderLayer(encoder_self_attention , encoder_feedforward , dropout)
        encoder_layers.append(encoder_layer)
    decoder_layers = []
    for _ in range(N) :
        decoder_self_attention = MultiHeadAttentions(d_model , head , dropout)
        decoder_feedforward = FeedForwardNetwork(d_model , d_ff , dropout)
        decoder_cross_attention = MultiHeadAttentions(d_model , head , dropout)
        decoder_layer = DecoderLayer(decoder_self_attention , decoder_cross_attention , decoder_feedforward , dropout)
        decoder_layers.append(decoder_layer)

    encoder = Encoder(nn.ModuleList(encoder_layer))
    decoder = Decoder(nn.ModuleList(decoder_layer))
    generation_layer = Generation(d_model , trg_vocab)

    transformer = EncoderDecoder(encoder , decoder , src_embedding , trg_embedding , src_position , trg_position , generation_layer)

    for p in transformer.parameters() :
        if p.dim() > 1 :
            nn.init.xavier_uniform_(p)
    return transformer                

                           

    
                     