import torch
import torch.nn as nn
from torch.nn import functional as F

class PhraseAttention(nn.Module):
  def __init__(self, input_dim):
    super(PhraseAttention, self).__init__()
    # initialize pivot
    self.fc = nn.Linear(input_dim, 1)

  def forward(self, context, embedded, input_labels):
    """
    Inputs:
    - context : Variable float (batch, seq_len, input_dim)
    - embedded: Variable float (batch, seq_len, word_vec_size)
    - input_labels: Variable long (batch, seq_len)
    Outputs:
    - attn    : Variable float (batch, seq_len)
    - weighted_emb: Variable float (batch, word_vec_size)
    """

    # print(context.shape)  # [N, S, C]
    cxt_scores = self.fc(context).squeeze(2) # (batch, seq_len)
    attn = F.softmax(cxt_scores, dim=-1)  # (batch, seq_len), attn.sum(1) = 1.

    # mask zeros
    is_not_zero = (input_labels!=0).float() # (batch, seq_len)
    attn = attn * is_not_zero # (batch, seq_len)
    attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1)) # (batch, seq_len)

    # compute weighted embedding
    attn3 = attn.unsqueeze(1)     # (batch, 1, seq_len)
    weighted_emb = torch.bmm(attn3, embedded) # (batch, 1, word_vec_size)
    weighted_emb = weighted_emb.squeeze(1)    # (batch, word_vec_size)

    return attn, weighted_emb


class PhraseMatcher(nn.Module):
    def __init__(self, vis_dim, lang_dim, embed_dim, pro_dim, jemb_drop_out=0.1):
        super(PhraseMatcher, self).__init__()

        self.sub_match = nn.Sequential(
            nn.Conv2d(vis_dim, pro_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(pro_dim),
            nn.ReLU(),
            nn.Dropout(jemb_drop_out),
            nn.Conv2d(pro_dim, pro_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(pro_dim),
        )
        self.sub_lang_proj = nn.Sequential(
            nn.Linear(embed_dim, pro_dim),
            # nn.BatchNorm1d(pro_dim),
            nn.ReLU(),
            nn.Linear(pro_dim, pro_dim),
            # nn.BatchNorm1d(pro_dim)
            nn.ReLU()
        )
        self.sub_attn = PhraseAttention(lang_dim)
        
        self.rel_match = nn.Sequential(
            nn.Conv2d(vis_dim, pro_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(pro_dim),
            nn.ReLU(),
            nn.Dropout(jemb_drop_out),
            nn.Conv2d(pro_dim, pro_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(pro_dim),
        )
        self.rel_lang_proj = nn.Sequential(
            nn.Linear(embed_dim, pro_dim),
            # nn.BatchNorm1d(pro_dim),
            nn.ReLU(),
            nn.Linear(pro_dim, pro_dim),
            # nn.BatchNorm1d(pro_dim)
            nn.ReLU()
        )
        self.rel_attn = PhraseAttention(lang_dim)
        
        self.pos_match = nn.Sequential(
            nn.Conv2d(vis_dim, pro_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(pro_dim),
            nn.ReLU(),
            nn.Dropout(jemb_drop_out),
            nn.Conv2d(pro_dim, pro_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(pro_dim),
        )
        self.pos_lang_proj = nn.Sequential(
            nn.Linear(embed_dim, pro_dim),
            # nn.BatchNorm1d(pro_dim),
            nn.ReLU(),
            nn.Linear(pro_dim, pro_dim),
            # nn.BatchNorm1d(pro_dim)
            nn.ReLU()
        )
        self.pos_attn = PhraseAttention(lang_dim)

        self.back = nn.Conv2d(pro_dim, vis_dim, 1, bias=False)

    def forward(self, vis_feature, hs, hn, embedding, words):
        N, C, vh, vw = vis_feature.shape

        vis_sub = self.sub_match(vis_feature)  # [N, D, vh, vw]
        # sub_hs = self.sub_lang_proj(hs)
        attn_sub, weighted_sub_emb = self.sub_attn(hs, embedding, words)
        # print(vis_sub.shape)
        # print(sub_context.shape)  # 前后不改变维度
        sub_context = self.sub_lang_proj(weighted_sub_emb)
        sub_context = sub_context.reshape(N, -1, 1, 1).repeat(1, 1, vh, vw)

        sub_feature = torch.mul(vis_sub, sub_context)

        vis_rel = self.rel_match(vis_feature)  # [N, D, vh, vw]
        # rel_hs = self.sub_lang_proj(hs)
        attn_rel, weighted_rel_emb = self.rel_attn(hs, embedding, words)
        rel_context = self.rel_lang_proj(weighted_rel_emb)
        rel_context = rel_context.reshape(N, -1, 1, 1).repeat(1, 1, vh, vw)
        rel_feature = torch.mul(vis_rel, rel_context)

        vis_pos = self.pos_match(vis_feature)  # [N, D, vh, vw]
        # pos_hs = self.sub_lang_proj(hs)
        attn_pos, weighted_pos_emb = self.pos_attn(hs, embedding, words)
        pos_context = self.pos_lang_proj(weighted_pos_emb)
        pos_context = pos_context.reshape(N, -1, 1, 1).repeat(1, 1, vh, vw)
        pos_feature = torch.mul(vis_pos, pos_context)


        attn_feature = sub_feature + rel_feature + pos_feature
        attn_feature = F.normalize(attn_feature, p=2, dim=1)

        attn_feature = self.back(attn_feature)

        # print(attn_feature.shape)

        # print(weighted_sub_emb.shape)
        # print(weighted_rel_emb.shape)
        # print(weighted_pos_emb.shape)
        #  like hn.shape: [N, dim]
        # assert 1 == 0

        return attn_feature


class MATTN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("使用recurrent注意力机制")
        self.rnn_dim = cfg.REF_RNN_DIM
        bidirectional = 2
        self.hn_dim = self.rnn_dim * bidirectional
        self.hs_dim = cfg.REF_RNN_DIM * bidirectional
        self.embedding_dim = cfg.WORD_VEC_DIM  
        self.visual_dim = 256
        self.attn_dim = self.hs_dim + self.embedding_dim

        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.acti1 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d((3, 3))
        self.acti2 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool2d((6, 6))
        self.acti3 = nn.ReLU()
        self.pool4 = nn.AdaptiveAvgPool2d((8, 8))
        self.acti4 = nn.ReLU()

        self.m_proj1 = nn.Conv2d(self.visual_dim, self.visual_dim, 1, bias=False)
        self.m_proj2 = nn.Conv2d(self.visual_dim, self.visual_dim, 1, bias=False) 
        self.m_proj3 = nn.Conv2d(self.visual_dim, self.visual_dim, 1, bias=False)

        self.l_proj1 = nn.Conv2d(self.visual_dim, self.visual_dim, 1, bias=False)
        self.l_proj2 = nn.Conv2d(self.visual_dim, self.visual_dim, 1, bias=False)
        self.l_proj3 = nn.Conv2d(self.visual_dim, self.visual_dim, 1, bias=False)


        self.fusion = nn.Conv2d(self.visual_dim*2, self.visual_dim, 3, bias=False)

    def sample(self, f):
        N, C = f.shape[0], f.shape[1]

        f1 = self.acti1(self.pool1(f)) # [N, C, 1, 1]
        f1 = f1.reshape(N, C, -1)
        
        f2 = self.acti2(self.pool2(f)) # [N, C, 3, 3]
        f2 = f2.reshape(N, C, -1)
        
        f3 = self.acti3(self.pool3(f)) # [N, C, 6, 6]
        f3 = f3.reshape(N, C, -1)
        
        f4 = self.acti4(self.pool4(f)) # [N, C, 8, 8]
        f4 = f4.reshape(N, C, -1)

        f = torch.cat((f1, f2, f3, f4), dim=-1)  # [N, C, 1*1+3*3+6*6+8*8]
        f = F.normalize(f, p=2, dim=-1)

        return f    

    def forward(self, x1, x2):
        N, C, h, w = x1.shape
        assert x1.shape == x2.shape

        M1 = self.m_proj1(x1)
        M2 = self.m_proj2(x1).reshape(N, C, -1)  # [N, C, h*w]
        M3 = self.m_proj3(x1)

        SAM = self.sample(M1).permute(0, 2, 1)
        # print(SAM.shape) # [N, C, M]

        SAM_ = torch.bmm(SAM, M2) # [N, M, h*w]

        L1 = self.l_proj1(x2)
        L2 = self.l_proj2(x2).reshape(N, C, -1)
        L3 = self.l_proj3(x2)

        SAL = self.sample(L1).permute(0, 2, 1)
        # print(SAL.shape) # [N, C, M]

        SAL_ = torch.bmm(SAL, L2) # [N, M, h*w]
        # print(SAL.shape)
        A = torch.softmax((SAM_ + SAL_).permute(0, 2, 1), dim=-1)  # [N, h*w, M]
        # print(A.shape)
        M_ = torch.bmm(A, self.sample(M3).permute(0, 2, 1))  # [N, h*w, C]
        L_ = torch.bmm(A, self.sample(L3).permute(0, 2, 1))  # [N, h*w, C]


        M_ = M_.permute(0, 2, 1).reshape(N, C, h, w)
        L_ = L_.permute(0, 2, 1).reshape(N, C, h, w)

        fea = torch.cat((M_, L_), dim=1)
        fea = F.normalize(fea, p=2, dim=1)

        fea = self.fusion(fea)

        return F.relu(fea)


class AsyCA(nn.Module):
    def __init__(self, num_features, ratio):
        super(AsyCA, self).__init__()
        self.out_channels = num_features
        self.conv_init = nn.Conv2d(num_features * 2, num_features, kernel_size=1, padding=0, stride=1)
        self.conv_dc = nn.Conv2d(num_features, num_features // ratio, kernel_size=1, padding=0, stride=1)
        self.conv_ic = nn.Conv2d(num_features // ratio, num_features * 2, kernel_size=1, padding=0, stride=1)
        self.act = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2): # 一般来说，x1可以是
        batch_size = x1.size(0)
        assert x1.shape[1] == x2.shape[1]
        
        feat_init = torch.cat((x1, x2), 1)  #  [N, d1+d2, vh, vw]
        feat_init = self.conv_init(feat_init)  # [N, d, vh, vw]
        fea_avg = self.avg_pool(feat_init) # [N, d, 1, 1]
        feat_ca = self.conv_dc(fea_avg)  # [N, d//2, 1, 1]
        feat_ca = self.conv_ic(self.act(feat_ca))  # [N, 2*d, 1, 1]

        a_b = feat_ca.reshape(batch_size, 2, self.out_channels, -1)  # [N, 2, d, 1]
        # print(a_b.shape)
        a_b = self.softmax(a_b)  
        a_b = list(a_b.chunk(2, dim=1))  # split to a and b  # [N, d, 1] 两份
        a_b = list(map(lambda x1: x1.reshape(batch_size, self.out_channels, 1, 1), a_b)) # [N, d, vh, vw]
        V1 = a_b[0] * x1
        V2 = a_b[1] * x2
        V = V1 + V2
        
        return V

class GenerateRTTN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("使用transformer注意力机制")

        self.rnn_dim = cfg.REF_RNN_DIM
        bidirectional = 2
        self.hn_dim = self.rnn_dim * bidirectional
        self.hs_dim = cfg.REF_RNN_DIM * bidirectional
        self.embedding_dim = cfg.WORD_VEC_DIM  
        self.visual_dim = 256

        self.qconv = nn.Linear(self.embedding_dim, self.visual_dim)
        self.qacti = nn.ReLU()

        self.mconv = nn.Conv2d(self.visual_dim+8, self.visual_dim, 3, padding=1, bias=False)
        self.mnorm = nn.LayerNorm(self.visual_dim)
        self.macti = nn.ReLU()

    def forward(self, hs, hn, feature, embedding, words):
        N, C, vh, vw = feature.shape
        seq_len = hs.shape[1] 

        hn = hn.reshape(N, -1, 1, 1).repeat(1, 1, vh, vw)
        spatial = torch.as_tensor(generate_spatial_batch(N, vh, vw), dtype=torch.float32, device=feature.device) #[batch, vh, vw, 8]
        spatial = spatial.permute(0, 3, 1, 2)

        for n in range(seq_len):
            if words[0, n] == 0:
                continue

            embed_n = embedding[:, n, :]  # [N, d2]
            # q = torch.cat((h_n, embed_n), dim=1)
            q = embed_n
            q = self.qconv(q)
            q = self.qacti(q)   # [N, dim]

            tfeature = torch.cat((feature, spatial), dim=1)
            tfeature = self.mconv(tfeature)
            tfeature = self.macti(tfeature)
            tfeature = tfeature.reshape(N, -1, vh*vw)

            # q = q.reshape(N, 1, -1)

            attn_map = torch.bmm(q.reshape(N, 1, -1), tfeature)  # [N, 1, vh*vw]

            attn = torch.bmm(q.reshape(N, -1, 1), attn_map).reshape(N, -1, vh, vw)
            # print(attn.shape)
            # print(feature.shape)
            feature = feature + attn

        return feature

class TRTTN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("使用transformer注意力机制")

        self.rnn_dim = cfg.REF_RNN_DIM
        bidirectional = 2
        self.hn_dim = self.rnn_dim * bidirectional
        self.hs_dim = cfg.REF_RNN_DIM * bidirectional
        self.embedding_dim = cfg.WORD_VEC_DIM  
        self.visual_dim = 256
        self.attn_dim = self.hs_dim + self.embedding_dim

        self.qconv = nn.Linear(self.embedding_dim, self.visual_dim)
        self.qacti = nn.ReLU()

        self.mconv = nn.Conv2d(self.visual_dim+8+self.hn_dim, self.visual_dim, 3, padding=1, bias=False)
        self.mnorm = nn.LayerNorm(self.visual_dim)
        self.macti = nn.ReLU()

        self.head_num = 2
        self.self_attn = nn.MultiheadAttention(self.visual_dim, num_heads=self.head_num)
        dropout = 0.1
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.visual_dim)

        # FFN
        self.linear1_ffn = nn.Linear(self.visual_dim, self.visual_dim)
        self.acti_ffn = nn.ReLU()
        self.dropout1_ffn = nn.Dropout(dropout)
        self.linear2_ffn = nn.Linear(self.visual_dim, self.visual_dim)
        self.dropout2_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(self.visual_dim)



    def forward(self, hs, hn, feature, embedding, words):
        N, C, vh, vw = feature.shape
        seq_len = hs.shape[1] 

        hn = hn.reshape(N, -1, 1, 1).repeat(1, 1, vh, vw)
        spatial = torch.as_tensor(generate_spatial_batch(N, vh, vw), dtype=torch.float32, device=feature.device) #[batch, vh, vw, 8]
        spatial = spatial.permute(0, 3, 1, 2)

        for n in range(seq_len):
            if words[0, n] == 0:
                continue

            embed_n = embedding[:, n, :]  # [N, d2]
            # q = torch.cat((h_n, embed_n), dim=1)
            q = embed_n
            q = self.qconv(q)
            q = self.qacti(q)   # [N, dim]

            Q = q.reshape(N, 1, -1).repeat(1, vh*vw, 1) # [N, ON, C]  # 将
            tfeature = F.normalize(feature, p=2, dim=1)
            tfeature = torch.cat((feature, spatial, hn), dim=1)
            tfeature = self.mconv(tfeature)
            tfeature = self.macti(tfeature)
            tfeature = tfeature.permute(0, 2, 3, 1).reshape(N, vh*vw, -1)
            tfeature = self.mnorm(tfeature)

            V = K = tfeature
            # 注意力模块
            fea2 = self.self_attn(Q, K, value=V)[0]
            fea = V + self.dropout(fea2)
            fea = self.norm(fea)
            # FFN
            fea2 = self.linear2_ffn(self.dropout1_ffn(self.acti_ffn(self.linear1_ffn(fea))))
            fea = fea + self.dropout2_ffn(fea2)
            fea = self.norm_ffn(fea)

            feature = fea.permute(0, 2, 1).reshape(N, -1, vh, vw) 

        return feature
  
class RMIATTN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("使用rmi注意力机制")
        self.rnn_dim = cfg.REF_RNN_DIM
        bidirectional = 2
        self.hn_dim = self.rnn_dim * bidirectional
        self.hs_dim = cfg.REF_RNN_DIM * bidirectional
        self.embedding_dim = cfg.WORD_VEC_DIM  
        self.visual_dim = 256
        self.attn_dim = self.hs_dim + self.embedding_dim

        concat_featrue_dim = self.visual_dim + 8 + self.hs_dim + self.embedding_dim
        self.rmi = RMI(concat_featrue_dim, self.rnn_dim, rnn_num_layers=1)


        
    def forward(self, x, embedding, word_hidden_state):

        res = self.rmi(x, embedding, word_hidden_state)

        # attn = self.attn_conv(res)
        # attn = self.attn_acti(attn)
        # # attn = self.attn_bn(attn)
        # attn = self.sigmoid(attn)

        # # 添加一个attention优化器
        # # bres = self.bbn(res)
        # bres = self.bconv1(res)

        # return attn, bres
        return res

class RATTN(nn.Module):
    def __init__(self, rnn_dim, embedding_dim):
        super().__init__()
        print("使用recurrent注意力机制")
        self.rnn_dim = rnn_dim
        bidirectional = 2
        self.hn_dim = self.rnn_dim * bidirectional
        self.hs_dim = rnn_dim * bidirectional
        self.embedding_dim = embedding_dim  
        self.visual_dim = 256
        self.attn_dim = self.hs_dim

        self.qconv = nn.Linear(self.hs_dim, self.attn_dim)
        self.qacti = nn.Tanh()

        self.pro_fc = nn.Linear(self.visual_dim+8, self.attn_dim)

        # self.rnn = nn.LSTM(self.hs_dim, self.visual_dim, 1, dropout=0, batch_first=True)
        self.rnn = nn.LSTMCell(self.attn_dim, self.visual_dim)
        # self.rnn = nn.RNNCell(self.attn_dim, self.visual_dim)

        self.attn_conv = nn.Conv2d(self.visual_dim, self.visual_dim//2, kernel_size=3, padding=1, bias=False)
        self.attn_acti = nn.ReLU()
        self.out_attn = nn.Conv2d(self.visual_dim//2, 1, kernel_size=1, padding=0, bias=False)
        # self.attn_bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

        self.bconv = nn.Conv2d(self.visual_dim, self.visual_dim//2, kernel_size=3, padding=1, bias=False)
        self.bbn = nn.BatchNorm2d(self.visual_dim)
        self.bacti = nn.ReLU(inplace=True)
        self.out_bres = nn.Conv2d(self.visual_dim//2, 2, kernel_size=1, padding=0, bias=False)

        self.tmapping = nn.Linear(self.hs_dim+self.embedding_dim, self.visual_dim)
        self.trelu = nn.ReLU()
        self.vmapping = nn.Linear(self.visual_dim+8, self.visual_dim)
        self.vrelu = nn.ReLU()

        self.ssoftmax = nn.Softmax(dim=-1)

    def forward(self, hs, hn, feature, embedding, words):
        # hs: [N, seq_len, dim1]
        # feature: [N, dim2, vh, vw]
        # words: [N, seq_len, seq_len]
        N, _, vh, vw = feature.shape
        seq_len = hs.shape[1] 

        tfeature = F.normalize(feature, p=2, dim=1)
        spatial = torch.as_tensor(generate_spatial_batch(N, vh, vw), dtype=torch.float32, device=feature.device) #[batch, vh, vw, 8]
        spatial = spatial.reshape(-1, 8)
        # tfeature = torch.cat((tfeature, spatial), dim=1)
        tfeature = tfeature.permute(0, 2, 3, 1)  # [N, vh, vw, visual_dim]

        tfeature = tfeature.reshape(N*vh*vw, -1) 
        
        # print(hs.shape, embedding.shape)
        # hs = torch.cat((hs, embedding), dim=2)
        hs = hs.permute(0, 2, 1)  # [N, hs_dim, seq_len]
        hs = hs.reshape(N, 1, 1, -1, seq_len).repeat(1, vh, vw, 1, 1)  # [N, vh, vw, dim, seq_len]
        hs = hs.reshape(N*vh*vw, -1, seq_len)  ## [N*vh*vw, dim, seq_len]
        # hs = self.qconv(hs)
        # hs = self.qacti(hs)

        c = torch.zeros_like(tfeature)  # 初始上一个状态
        # c = Variable(c)
        h = tfeature  # 初始状态

        def f(ths, th):
            # print("attention")
            th = torch.cat((th, spatial), dim=1)
            th = self.pro_fc(th)  # [N*vh*vw, self.attn_dim]
            th = th.reshape(-1, 1, self.attn_dim) # [N*vh*vw, 1, self.attn_dim]

            attn_map = torch.bmm(th, ths)  # [N*vh*vw, 1, seq_len]
            attn_map = F.softmax(attn_map, dim=-1)  

            attn_feat = torch.bmm(attn_map, ths.permute(0, 2, 1))  # [N*vh*vw, 1, hs_dim]
            attn_feat = attn_feat.squeeze(1)  # [N*vh*vw, hs_dim]
            # print(attn_feat.shape)
            return attn_feat
            # _, t_hn, _ = self.rnn(attn_feat)  # [N*vh*vw, visual_dim]

        for n in range(words.shape[1]):
            if words[0, n] == 0:
                # print("无意义词")
                continue
            
            attn_feat = f(hs, h)
            h, c = self.rnn(attn_feat, (h, c))

        res = h.reshape(N, vh, vw, -1).permute(0, 3, 1, 2)  # [N, visual_dim, vh, vw]

        assert res.shape[1] == self.visual_dim
        # res = F.normalize(res, p=2, dim=1)

        # # 加个自注意
        # q = res.reshape(N, -1, vh*vw).permute(0, 2, 1) # [N, vh*vw, C]
        # k = res.reshape(N, -1, vh*vw)

        # attn = torch.bmm(q, k)  # [N, vh*vw, vh*vw]
        # attn = self.ssoftmax(attn)

        # attn_feat = torch.bmm(attn, res.reshape(N, -1, vh*vw).permute(0, 2, 1))
        # attn_feat = attn_feat.permute(0, 2, 1).reshape(N, -1, vh, vw)
        # res = res + attn_feat

        # res = res + feature #添加一个残差
        # attn = self.attn_conv(res)
        # attn = self.attn_acti(attn)
        # attn = self.out_attn(attn)
        # # attn = self.attn_bn(attn)
        # attn = self.sigmoid(attn)

        # # # 添加一个attention优化器
        # bres = self.bconv(res)
        # bres = self.bacti(bres)
        # bres = self.out_bres(bres)

        # return attn, bres
        # return res + feature
        return res



from ..utils.comm import generate_spatial_batch 
class RMI(nn.Module):
    def __init__(self, concat_featrue_dim=1000, rnn_dim=256, rnn_num_layers=1, rnn_dropout=0, use_attn=False):
        super(RMI, self).__init__()
        print("使用RMI")

        self.rnn = nn.LSTM(concat_featrue_dim, rnn_dim, rnn_num_layers, dropout=rnn_dropout, batch_first=True)
        
    
    def forward(self, x, embedding, word_hidden_state):

        vh, vw = x.shape[-2: ]
        batch_size = x.shape[0]
        seq_len = embedding.shape[1]
        hidden_size = word_hidden_state.shape[-1]

        # 把这个计算位置换一下
        spatial = torch.as_tensor(generate_spatial_batch(batch_size, vh, vw), dtype=torch.float32) #[batch, vh, vw, 8]
        # spatial = torch.as_tensor(generate_coord(batch_size, vh, vw), dtype=torch.float32)
        # spatial = torch.as_tensor(g2(batch_size, vh, vw, stride), dtype=torch.float32, device=feature.device)
        
        # spatial = spatial.permute(0, 3, 1, 2)  # [batch, 8, vh, vw]
        # 5. make use of the hs, hs include the hidden state of all time step (all word)
        # hs: [batch, seq_len, hidden_size * bidirect]
        # 
        # hs = word_hidden_state.permute(0, 2, 1) # [batch, hidden_size * bidirect, seq_len]
        hs = F.normalize(word_hidden_state, p=2, dim=2) # [batch, seq_len, hidden_size * bidirect]

        # 自注意力好像应该加到这个上面
        # if self.use_attn:
        #     hs = hs.permute(1, 0, 2)  # [seq_len, N, hidden_size]
        #     hs2 = self.multi_attn(hs, hs, value=hs)[0]
            # hs = hs + self.dropout(hs2)
            # hs = self.norm(hs) # [seq_len, N, hidden_size]
            # hs = hs.permute(1, 0, 2) # [N, seq_len, hidden_size]



        hs = torch.reshape(hs, (batch_size, 1, 1, seq_len, hs.shape[2]))  # [batch, 1, 1, seq_len, hidden_size * bidirect]
        hs = hs.repeat(1, vh, vw, 1, 1) # [batch, vh, vw, seq_len, hidden_size * bidirect]

        # 6. make use of the word embedding
        # embedding = embedding.transpose(1, 2) # [batch, word_vec_dim, seq_len]
        embedding = torch.reshape(embedding, (batch_size, 1, 1, seq_len, embedding.shape[2])) # [batch, 1, 1, seq_len, word_vec_dim]
        embedding = embedding.repeat(1, vh, vw, 1, 1) # [batch, vh, vw, seq_len, word_vec_dim]

        # 7. make use of the visual embedding
        x = x.permute(0, 2, 3, 1)  # [N, vh, vw, visual_embed_dim]
        x = torch.reshape(x, (batch_size, vh, vw, 1, -1))  # [N, vh, vw, 1, visual_embed_dim]
        x = x.repeat(1, 1, 1, seq_len, 1) # [N, vh, vw, seq_len, visual_embed_dim]

        # 8. make use of the spatial coord
        spatial = torch.reshape(spatial, (batch_size, vh, vw, 1, -1))
        spatial = spatial.repeat(1, 1, 1, seq_len, 1)  # [batch, vh, vw, seq_len, 8]
        # concat
        # before concat, should keep all data on gpu or cpu
        if x.is_cuda:
            spatial = spatial.to(x.device)
            embedding = embedding.to(x.device)
            hs = hs.to(x.device)
        
        
        
        feat_concat = torch.cat((hs, embedding, x, spatial), dim=-1) # [batch, vh, vw, seq_len, hidden_size + word_vec_dim + visual_embed_dim + 8]  
        # feat_concat = feat_concat.transpose(1, 2)
        feat_concat = torch.reshape(feat_concat, (batch_size * vh * vw, seq_len, -1)) # [batch*vh*vw, seq_len, hidden_size + word_vec_dim + visual_embed_dim + 8]


        self.rnn.flatten_parameters()
        # print(feat_concat.shape)
        # print(self.rnn)
        hs, (hn, cn) = self.rnn(feat_concat)
        # hs: [batch, max_Len, hidden_size]

        hn, cn = hn.transpose(0, 1).contiguous(), cn.transpose(0, 1).contiguous()  # hn: [batch, num_layer * bidirect, hiddens_size], cn: [batch, num_layer * bidirect, hiddens_size]
        hn, cn = hn.view(hn.size(0), -1), cn.view(cn.size(0), -1)  #[batch, num_layer * bidirect * hiddens_size]

        # hs: [N*vh*vw, seq_len, hidden_size]
        # hn: [N*vh*vw, hidden_size]
        # 
        
        # if self.use_attn:
        #     hn = hn.reshape(batch_size, -1, hidden_size)  # [N, vh*vw, hidden_size]
        #     hn = hn.permute(1, 0, 2)   # [vh*vw, N, hidden_size]
        #     hn2 = self.multi_attn(hn, hn, value=hn)[0]
        #     hn = hn + self.dropout(hn2)
            # hn = self.norm(hn)  # [vh*vw, N, hidden_size]
            # hn = hn.reshape(-1, hidden_size)  #[N*vh*vw, hidden_size]
            
        hn = hn.reshape(batch_size, vh, vw, -1).permute(0, 3, 1, 2)
        return hn
