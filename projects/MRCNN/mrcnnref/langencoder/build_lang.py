import os
from .rnn import RNNEncoder, LstmEncoder
from ..utils.comm import load_vocab

def init_ref(cfg, is_training=True):
        rnn_dim = cfg.REF_RNN_DIM
        num_layer = cfg.NUM_HIDEEN_LAYER
        word_vec_dim = cfg.WORD_VEC_DIM

        # iep-ref
        # vocab_file = "/home/lingpeng/project/iep-ref-master/data/referring_rubber/picked_vocab_train.json"
        # refcoco
        # self.vocab_file = os.path.join("/home/lingpeng/project/demo/data/", "refcoco", "picked_vocab_train.json")
        # vocab_file = os.path.join("/nfs/crefs/", "dict", "refcoco", 'picked_c_vocab.json')

        if cfg.MODEL.NUM_CLASSES == 46:
        	vocab_file = "/nfs/SketchySceneColorization/Instance_Matching/data/vocab.json"
        
        elif cfg.MODEL.NUM_CLASSES == 90:
            vocab_file = "/nfs/demo/data/refcoco/vocab.json"
            
        else:
            vocab_file = "/nfs/crefs/dict/phrasecut/dict.json"

        vocab_size = len(load_vocab(vocab_file)["refexp_token_to_idx"])
        # word_embedding_size = cfg.WORD_VEC_DIM 
        word_embedding_size = 256
        
        textencoder = RNNEncoder(vocab_size, word_embedding_size, word_vec_dim, rnn_dim, bidirectional=True, rnn_num_layers=num_layer, is_train=is_training)
        # textencoder = LstmEncoder(load_vocab(vocab_file)["refexp_token_to_idx"], 
                                        # rnn_dim=rnn_dim, wordvec_dim=word_vec_dim, rnn_num_layers=num_layer)

        return textencoder