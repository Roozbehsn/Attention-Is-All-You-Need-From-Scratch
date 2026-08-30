from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from torch.nn import Embedding
from torch import nn
import torch
import math

input = "I love cats"



class LayerNormalization(nn.Module) :
    def __init__(self , eps=1e-6):
        super().__init__()
        self.gamma = nn.parameter(torch.ones(1))
        self.beta = nn.parameter(torch.zeros(1))
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
        return self.w2(self.droput(torch.ReLU(self.w1(x))))


class LayerNormalization(nn.Module) :
    def __init__(self , eps=1e-6):
        super().__init__()
        self.gamma = nn.parameter(torch.ones(1))
        self.beta = nn.parameter(torch.zeros(1))
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
        return self.w2(self.droput(torch.ReLU(self.w1(x))))

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

    def attention(query , key , value , mask , dropout):
        d_k = query.size(-1)
        attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None :
            attention_score.masked_fill_(mask == 0 , -1e9)
        attention_score = attention_score.softmax(dim = -1)
        if dropout is None :
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
        x = x.transpose(1,2).contiguous().view(x.shape[0] , 1 , self.d_k * self.head)

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
                           

    
                     