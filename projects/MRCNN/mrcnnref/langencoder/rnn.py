import numpy as np
import torch.nn as nn
import torch
from ..utils.comm import FeatureResizer



class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, wordvec_dim=300,
               rnn_dim=256, bidirectional=False, rnn_num_layers=2, rnn_dropout=0.1, cfg_sys=None, is_train=True):
        super(RNNEncoder, self).__init__()

        self.variable_length      = True
        self.word_embedding_size  = word_embedding_size
        self.word_vec_size        = wordvec_dim
        self.hidden_size          = rnn_dim
        self.bidirectional        = bidirectional
        self.input_dropout_p      = rnn_dropout
        # self.dropout_p            = rnn_dropout
        self.dropout_p            = rnn_dropout * (rnn_num_layers - 1)
        self.n_layers             = rnn_num_layers
        self.rnn_type             = 'lstm' # by default LSTM
        self.vocab_size = vocab_size
        self.is_train = is_train

        # encoder language
        self.embedding      = nn.Embedding(self.vocab_size, self.word_embedding_size)
        self.input_dropout  = nn.Dropout(self.input_dropout_p)
        self.mlp            = nn.Sequential(nn.Linear(self.word_embedding_size, self.word_vec_size), nn.ReLU())
        self.rnn            = getattr(nn, self.rnn_type.upper())(self.word_vec_size, 
                                                            self.hidden_size, 
                                                            self.n_layers,
                                                            batch_first=True,
                                                            bidirectional=self.bidirectional,
                                                            dropout = self.dropout_p)
        self.num_dirs = 2 if self.bidirectional else 1

    def forward(self, input, mask=None):
        word_id = input
        max_len = (word_id!=0).sum(1).max().item()
        word_id = word_id[:, :max_len] # mask zero
        # embedding
        output, hidden, embedded, final_output = self.RNNEncode(word_id)
        return {
            'hidden': hidden,
            'output': output,
            'embedded': embedded,
            'final_output': final_output,
        }

    def RNNEncode(self, input_labels):
        # print(input_labels)
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len)
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """
        device = input_labels.device
        if self.variable_length:
            input_lengths_list, sorted_lengths_list, sort_idxs, recover_idxs = self.sort_inputs(input_labels)
            input_labels = input_labels[sort_idxs]
        
        embedded = self.embedding(input_labels) #(n, seq_len, word_embedding_size)
        # if self.is_train:
        embedded = self.input_dropout(embedded) #(n, seq_len, word_embedding_size)
            
        # print(embedded.shape)
        # print(embedded)
        embedded = self.mlp(embedded)           #(n, seq_len, word_vec_size)

        if self.variable_length:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, \
                                                        sorted_lengths_list,\
                                                         batch_first=True)
        # forward rnn
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(embedded)
        
        # recover
        if self.variable_length:
            # recover embedded
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)  # (batch, max_len, word_vec_size)
            embedded = embedded[recover_idxs]

            # recover output
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)      # (batch, max_len, hidden_size * num_dir)
            output = output[recover_idxs]

            # recover hidden
            if self.rnn_type == 'lstm':
                hidden = hidden[0]                      # hidden state
            hidden = hidden[:, recover_idxs, :]         # (num_layers * num_dirs, batch, hidden_size)
            hidden = hidden.transpose(0,1).contiguous() # (batch, num_layers * num_dirs, hidden_size)
            hidden = hidden.view(hidden.size(0), -1)    # (batch, num_layers * num_dirs * hidden_size)
        
        # finnal output
        finnal_output = []
        for ii in range(output.shape[0]):
            finnal_output.append(output[ii, int(input_lengths_list[ii]-1), :])
        finnal_output = torch.stack(finnal_output, dim=0)   # (batch, number_dirs * hidden_size)

        return output, hidden, embedded, finnal_output

    def sort_inputs(self, input_labels):                                                # sort input labels by descending
        device = input_labels.device
        input_lengths = (input_labels!=0).sum(1)
        input_lengths_list = input_lengths.data.cpu().numpy().tolist()
        sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()          # list of sorted input_lengths
        sort_idxs = np.argsort(input_lengths_list)[::-1].tolist()
        s2r = {s:r for r, s in enumerate(sort_idxs)}
        recover_idxs = [s2r[s] for s in range(len(input_lengths_list))]
        assert max(input_lengths_list) == input_labels.size(1)
        # move to long tensor
        sort_idxs = input_labels.data.new(sort_idxs).long().to(device)             # Variable long
        recover_idxs = input_labels.data.new(recover_idxs).long().to(device)       # Variable long
        return input_lengths_list, sorted_input_lengths_list, sort_idxs, recover_idxs



from .embedding import expand_embedding_vocab
from torch.autograd import Variable

class LstmEncoder(nn.Module):
  def __init__(self, token_to_idx, wordvec_dim=300,
               rnn_dim=256, rnn_num_layers=2, rnn_dropout=0.1):
    super(LstmEncoder, self).__init__()
    self.token_to_idx = token_to_idx
    self.NULL = token_to_idx['<NULL>']
    self.START = token_to_idx['<START>']
    self.END = token_to_idx['<END>']

    bidirectional = 1
    self.word_embedding_size = 128

    self.embed = nn.Embedding(len(token_to_idx), self.word_embedding_size)

    self.mlp = nn.Sequential(nn.Linear(self.word_embedding_size, wordvec_dim), 
                                 nn.ReLU())
    self.rnn = nn.LSTM(wordvec_dim, rnn_dim, rnn_num_layers,
                       dropout=rnn_dropout, batch_first=True, bidirectional=(bidirectional == 1))

  def expand_vocab(self, token_to_idx, word2vec=None, std=0.01):
    expand_embedding_vocab(self.embed, token_to_idx,
                           word2vec=word2vec, std=std)

  def forward(self, x):
    # print(x.shape)
    N, T = x.size()  # [N, seq_len] seq_len固定为max_len
    idx = torch.LongTensor(N).fill_(T - 1)

    # Find the last non-null element in each sequence
    x_cpu = x.data.cpu()
    for i in range(N):
      for t in range(T - 1):
        if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
          idx[i] = t
          break
    idx = idx.type_as(x.data).long()
    idx = Variable(idx, requires_grad=False)

    embedding = self.embed(x)
    embedding = self.mlp(embedding)

    self.rnn.flatten_parameters()
    hs, (hn, cn) = self.rnn(embedding)
    idx = idx.view(N, 1, 1).expand(N, 1, hs.size(2))
    H = hs.size(2)
    # print(hs.gather(1, idx).view(N, H).shape)
    # assert 1 == 0
    # print(hs.shape)
    #此时的维度：hs: [batch, seq_len, hidden_size * bidirect], hn: [num_layer * bidirect, batch, hiddens_size], cn: [num_layer * bidirect, batch, hiddens_size]
    # 包含两层信息
    hn, cn = hn.transpose(0, 1).contiguous(), cn.transpose(0, 1).contiguous()  # hn: [batch, num_layer * bidirect, hiddens_size], cn: [batch, num_layer * bidirect, hiddens_size]
    hn, cn = hn.view(hn.size(0), -1), cn.view(cn.size(0), -1)  #[batch, num_layer * bidirect * hiddens_size]
    
    return hs, hn, embedding
    # return hs, hs.gather(1, idx).view(N, H), embedding

# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertModel, RobertaModel
from transformers import logging
logging.set_verbosity_error()


class CustomerRobertaModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("construct Roberta model")
        self.robert = RobertaModel.from_pretrained("roberta-base")
        # 也可以微调，但是现在估计跑不动
        for param in self.robert.parameters():
            param.requires_grad = False
        
        self.bert_dim = 768
        self.hn_dim = 1024
        self.resizer = FeatureResizer(
            self.bert_dim,
            self.hn_dim,
            0.1
        )   
            
    def forward(self, words, attention_mask=None):
        outputs = self.robert(words, attention_mask=attention_mask)
        
        pooled = outputs.pooler_output # hn , [N, 768]
        
        word_embedding = outputs.last_hidden_state  # [N, max_len, D]
        
        return word_embedding, pooled

class CustomerBert(nn.Module):

    def __init__(self):
        super().__init__()
        print("construct bert model")
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 也可以微调，但是现在估计跑不动
        for param in self.bert.parameters():
            param.requires_grad = False
            
        
        self.bert_dim = 768
        self.hn_dim = 1024
        self.resizer = FeatureResizer(
            self.bert_dim,
            self.hn_dim,
            0.1
        )
        # self.flc = nn.Sequential(
        #     nn.Linear(self.bert_dim, self.hn_dim),
        #     # nn.BatchNorm1d(self.hn_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hn_dim, self.hn_dim),
        #     nn.ReLU(),
        # )
        
        # self.wordlc = nn.Sequential(
        #     nn.Linear(self.bert_dim, self.hn_dim),
        #     # nn.BatchNorm1d(self.hn_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hn_dim, self.hn_dim),
        #     nn.ReLU(),
        # )
    
    def forward(self, words, attention_mask=None):
        # print(words.shape, attention_mask.shape)
        outputs = self.bert(words, attention_mask=attention_mask)
        
        pooled = outputs.pooler_output # hn , [N, 768]
        
        # pooled = self.flc(pooled)
        pooled = self.resizer(pooled)
        
        # 再找到word embedding
        word_embedding = outputs.last_hidden_state  # [N, max_len, D]
        
        # print(word_embedding.shape, pooled.shape)
        # word_embedding = self.wordlc(word_embedding)
        # word_embedding_ = []
        # for i in range(word_embedding.shape[1]):
        #     word_embedding_.append(self.wordlc(word_embedding[:, i, :]).unsqueeze(1))
        
        # word_embedding = torch.cat(word_embedding_, dim=1)
        word_embedding = self.resizer(word_embedding)
        
        return word_embedding, pooled
    
    def forward1(self, words, token_type_ids=None, attention_mask=None):
        
        all_encoder_layers, pooled = self.bert(words, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=True)
        
        word_embeddings = all_encoder_layers[-1] # [N, max_len, 768]

        return word_embeddings, pooled  # [N, max_len, 768]， [N, 768]

    def forward2(self, words, token_type_ids=None, attention_mask=None):

        all_encoder_layers, _ = self.bert(words, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=True)

        ## Sentence feature at the first position [cls]
        raw_flang = (all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:] \
                + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:]) / 4
        
        ## fix bert during training
        raw_flang = raw_flang.detach()

        return raw_flang
  
from transformers import RobertaModel, RobertaTokenizerFast  
class CustomerRoberta(nn.Module):

    def __init__(self):
        super().__init__()

        # self.bert = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        
        for param in self.text_encoder.parameters(): # freeze
            param.requires_grad = False
        
        self.bert_dim = 768
        self.hn_dim = 512
        self.resizer = FeatureResizer(
            self.bert_dim,
            self.hn_dim,
            0.1
        )
        
            
    def forward(self, captions, device):  # tokens = tokenizer.batch_encode_plus(["man", "the person"], padding="longest", return_tensors="pt")
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            text_features = encoded_text.last_hidden_state 
            
            text_features = self.resizer(text_features)    
            text_sentence_features = encoded_text.pooler_output  
            text_sentence_features = self.resizer(text_sentence_features)  
            
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        
        return text_features, text_sentence_features
        

#####################################
########  from GLIP: https://github.com/microsoft/GLIP/blob/main/maskrcnn_benchmark/modeling/language_backbone/bert_model.py
#####################################

from copy import deepcopy
import numpy as np
import torch
from torch import nn

# from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel


class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super(BertEncoder, self).__init__()
        self.cfg = cfg
        self.bert_name = "bert-base-uncased"
        print("LANGUAGE BACKBONE USE GRADIENT CHECKPOINTING: ", False)

        if self.bert_name == "bert-base-uncased":
            config = BertConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = False
            self.model = BertModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        elif self.bert_name == "roberta-base":
            config = RobertaConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = False
            self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        else:
            raise NotImplementedError

        self.num_layers = 1

    def forward(self, x):
        input = x["input_ids"]
        mask = x["attention_mask"]

        if False:
            # with padding, always 256
            outputs = self.model(
                input_ids=input,
                attention_mask=mask,
                output_hidden_states=True,
            )
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]
            features = None
            features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)

            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.num_layers

            embedded = features * mask.unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        else:
            # without padding, only consider positive_tokens
            max_len = (input != 0).sum(1).max().item()
            outputs = self.model(
                input_ids=input[:, :max_len],
                attention_mask=mask[:, :max_len],
                output_hidden_states=True,
            )
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]

            features = None
            features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)
            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.num_layers

            embedded = features * mask[:, :max_len].unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        ret = {
            "aggregate": aggregate,
            "embedded": embedded,
            "masks": mask,
            "hidden": encoded_layers[-1]
        }
        return ret


import torch.nn.functional as F
class BertTextCNN(nn.Module):
    def __init__(self, hidden_size=256):
        super(BertTextCNN, self).__init__()
        # self.cfg = cfg
        self.bert_name = "bert-base-uncased"
        print("LANGUAGE BACKBONE USE GRADIENT CHECKPOINTING: ", False)

        if self.bert_name == "bert-base-uncased":
            config = BertConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = False
            self.model = BertModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        elif self.bert_name == "roberta-base":
            config = RobertaConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = False
            self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        else:
            raise NotImplementedError

        self.num_layers = 1

        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv2d(1, hidden_size, (3, 768))
        self.conv2 = nn.Conv2d(1, hidden_size, (4, 768))
        self.conv3 = nn.Conv2d(1, hidden_size, (5, 768))
        
        # for key, value in self.named_parameters(recurse=True):
        #     print(key)
            # if "encoder.layer.11" in key:
            #     print("第12层")
        
    def forward(self, x):
        input = x["input_ids"]
        mask = x["attention_mask"]
        
        max_len = (input != 0).sum(1).max().item()
        sequence_output = self.model(
            input_ids=input[:, :max_len],
            attention_mask=mask[:, :max_len],
            # output_all_encoded_layers=False # 只要最后一层的隐向量表示，其余11层不输出
        )
        sequence_output = sequence_output[-1]
        print(sequence_output.shape) # [N, sen_len, 768]
        out = self.dropout(sequence_output).unsqueeze(1) # [N, 1, sen_len, 768]
        c1 = torch.relu(self.conv1(out).squeeze(3))
        
        print(c1.shape) # [N, 128, sen_len-3+1]
        p1 = F.max_pool1d(c1, c1.size(2)).squeeze(2)
        print(p1.shape)
        c2 = torch.relu(self.conv2(out).squeeze(3))
        p2 = F.max_pool1d(c2, c2.size(2)).squeeze(2)
        c3 = torch.relu(self.conv3(out).squeeze(3))
        p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)
        pool = self.dropout(torch.cat((p1, p2, p3), 1))
        
        return pool
    


##################
# CLIP office
##################
from transformers import CLIPTextModel

class CLIPEncoder(nn.Module):
    def __init__(self):
        super(CLIPEncoder, self).__init__()
        # self.cfg = cfg
        self.model_name = "openai/clip-vit-base-patch32"
        print("build clip encoder: openai/clip-vit-base-patch32")

        self.model = CLIPTextModel.from_pretrained(self.model_name)

    
    def forward2(self, x):
        input = x["input_ids"]
        mask = x["attention_mask"]
        
        max_len = (input != 0).sum(1).max().item()
        outputs = self.model(
            input_ids=input[:, :max_len],
            attention_mask=mask[:, :max_len],
            output_hidden_states=True,
        )
        
        encoded_layers = outputs.hidden_states[1:]
        features = None
        # features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)  # only last
        features = outputs.last_hidden_state  # CLIP和BERT差异：CLIP的last_hidden会过最后一个LN，而hidden_states都不会；BERT中的last_hidden则就是最后一个hidden_states
        # features = torch.stack([encoded_layers[-1], encoded_layers[0]], 1).mean(1)  # first-last avg
        
        # language embedding has shape [len(phrase), seq_len, language_dim]
        # features = features / self.num_layers

        embedded = features * mask[:, :max_len].unsqueeze(-1).float()
        ret = {
            "aggregate": outputs.pooler_output,  # 查看源码发现，CLIP里面的pooler采取的是max pool的策略
            "embedded": embedded,
            "masks": mask,
            "hidden": encoded_layers[-1]
        }
        return ret
    
    def forward(self, x):
        input = x["input_ids"]
        mask = x["attention_mask"]
        
        max_len = (input != 0).sum(1).max().item()
        outputs = self.model(
            input_ids=input[:, :max_len],
            attention_mask=mask[:, :max_len],
            output_hidden_states=True,
        )
        # outputs has 13 layers, 1 input layer and 12 hidden layers
        # print(outputs.hidden_states)
        # assert 1 == 0
        encoded_layers = outputs.hidden_states[1:]

        features = None
        # features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)  # only last
        features = outputs.last_hidden_state  # CLIP和BERT差异：CLIP的last_hidden会过最后一个LN，而hidden_states都不会；BERT中的last_hidden则就是最后一个hidden_states
        # features = torch.stack([encoded_layers[-1], encoded_layers[0]], 1).mean(1)  # first-last avg
        
        # language embedding has shape [len(phrase), seq_len, language_dim]
        # features = features / self.num_layers

        embedded = features * mask[:, :max_len].unsqueeze(-1).float()
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        ret = {
            "aggregate": aggregate,
            "embedded": embedded,
            "masks": mask,
            "hidden": features
        }
        return ret
        
##################
#  CLIP non-office
#################
from collections import OrderedDict
from timm.models.layers import DropPath, trunc_normal_
import torch.utils.checkpoint as checkpoint
import os

def try_to_find(file, return_dir=False, search_path=['./DATASET', './OUTPUT', './data', './MODEL']):
    if not file:
        return file

    if file.startswith('catalog://'):
        return file

    DATASET_PATH = ['./']
    if 'DATASET' in os.environ:
        DATASET_PATH.append(os.environ['DATASET'])
    DATASET_PATH += search_path

    for path in DATASET_PATH:
        if os.path.exists(os.path.join(path, file)):
            if return_dir:
                return path
            else:
                return os.path.join(path, file)

    print('Cannot find {} in {}'.format(file, DATASET_PATH))
    exit(1)
    
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = x + self.drop_path(self.attention(self.ln_1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x



class CLIPTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
        self.use_checkpoint = False
        print("LANGUAGE BACKBONE USE GRADIENT CHECKPOINTING: ", False)
        
        self.context_length = 256
        self.width = 512
        self.layers = 12
        self.heads = 8
        self.drop_path = 0.0
        self.vocab_size = 49408
        
        self.token_embedding = nn.Embedding(self.vocab_size, self.width)

        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, self.width)
        )

        # attn_mask = self.build_attention_mask()
        attn_mask = None

        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.layers)]  # stochastic depth decay rule
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(self.width, self.heads, attn_mask, dpr[i])
                for i in range(self.layers)
            ]
        )

        self.ln_final = LayerNorm(self.width)

        trunc_normal_(self.positional_embedding, std=.02)
        # nn.init.normal_(self.token_embedding, std=.02)
        trunc_normal_(self.token_embedding.weight, std=.02)
        self.apply(self._init_weights)

        # loading pre-trained weight from our CLIP models
        if len(self.cfg.MODEL.LANGUAGE_BACKBONE.WEIGHT) > 0:
            assert 1 == 0
            self.init_weights(pretrained=try_to_find(self.cfg.MODEL.LANGUAGE_BACKBONE.WEIGHT),
                              pretrained_layers=['*'])
            
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)

    def resize_pos_embed_1d(self, posemb, shape_new):
        # rescale the grid of position embeddings when loading from state_dict
        ntok_old = posemb.shape[0]
        if ntok_old > 1:
            ntok_new = shape_new[0]
            posemb_grid = posemb.unsqueeze(dim=0).permute(0, 2, 1).unsqueeze(dim=-1)
            posemb_grid = F.interpolate(posemb_grid, size=[ntok_new, 1], mode='bilinear')
            posemb_grid = posemb_grid.squeeze(dim=-1).permute(0, 2, 1).squeeze(dim=0)
            posemb = posemb_grid
        return posemb

    def init_weights(self, pretrained="", pretrained_layers=[], verbose=False):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location="cpu")
            # logger.info(f'=> loading pretrained clip text model {pretrained}')
            model_dict = self.state_dict()

            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if k.startswith('text.') and k[5:] in model_dict.keys():
                        need_init_state_dict[k[5:]] = v

            # notice the context length now changes from 77 to 256, so we need to resize the positional embedding
            if "positional_embedding" in need_init_state_dict.keys():
                old_pos_embed = need_init_state_dict["positional_embedding"].float()
                new_pos_embed = self.resize_pos_embed_1d(old_pos_embed,
                                                         (self.context_length, old_pos_embed.shape[1]))
                need_init_state_dict["positional_embedding"] = new_pos_embed
                
            self.load_state_dict(need_init_state_dict, strict=True)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'positional_embedding',
            'token_embedding',
        }
    
    def forward(self, text):
        input = text["input_ids"]
        mask = text["attention_mask"]
        # get extended attention mask for nn.MultiHeadAttention
        key_padding_mask = (1.0 - mask).to(torch.bool)

        x = self.token_embedding(input)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND

        for resblock in self.resblocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(resblock, x, key_padding_mask)
            else:
                x = resblock(x, key_padding_mask)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x)

        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        ret = {
            "aggregate": x,
            "embedded": x,
            "masks": mask,
            "hidden": x
        }

        return ret