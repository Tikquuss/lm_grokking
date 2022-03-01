"""
code for vanilla Transformers (Attention is all you need) & BERT, with the option to use TIM (Transformer with Independent Mechanisms) layers
"""

import torch.nn as nn
import torch.nn.functional as F

import math
import itertools
import numpy as np
import torch

from .tim import TransformerEncoderLayer, TransformerDecoderLayer

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    # nn.init.normal_(m.weight, mean=0, std=1)
    # nn.init.xavier_uniform_(m.weight)
    # nn.init.constant_(m.bias, 0.)
    return m

def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False

def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask

def get_padding_masks(slen, lengths):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]
    # sanity check
    assert mask.size() == (bs, slen)
    return mask

def get_causal_masks(mask, slen):
    """
    Generate attention mask (triangular inferior attention).
    """
    bs = mask.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=mask.device)
    attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    # sanity check
    assert attn_mask.size() == (bs, slen, slen)
    return attn_mask

class PredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """
    def __init__(self, params):
        super().__init__()
        self.asm = getattr(params, "asm", False)
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        dim = params.emb_dim

        if self.asm is False:
            self.proj = Linear(dim, params.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=params.n_words,
                cutoffs=params.asm_cutoffs,
                div_value=params.asm_div_value,
                head_bias=True,  # default is False
            )

    def forward(self, x, y, reduction='mean'):
        """
        Compute the loss and the scores.
        """
        assert (y == self.pad_index).sum().item() == 0

        if self.asm is False:
            scores = self.proj(x).view(-1, self.n_words)
            loss = F.cross_entropy(scores, y, reduction=reduction)
        else:
            _, loss = self.proj(x, y)
            scores = self.proj.log_prob(x)

        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj.log_prob(x) if self.asm else self.proj(x)

class MultiHeadAttention(nn.Module):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.v_lin = Linear(dim, dim)
        self.out_lin = Linear(dim, dim)

    def forward(self, input, mask, kv=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        klen = qlen if kv is None else kv.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))                                          # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)

        q = q / math.sqrt(dim_per_head)                                       # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))                           # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)               # (bs, n_heads, qlen, klen)
        # to line is to solve :
        # /pytorch/aten/src/ATen/native/cuda/LegacyDefinitions.cpp:19: UserWarning: masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated,please use a mask with dtype torch.bool instead.
        mask = mask.bool()
        scores.masked_fill_(mask, -float('inf'))                              # (bs, n_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)           # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
        context = torch.matmul(weights, v)                                    # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)                                            # (bs, qlen, dim)

        return self.out_lin(context)

class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, dropout, gelu_activation):
        super().__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_dim, dim_hidden)
        self.lin2 = Linear(dim_hidden, out_dim)
        self.act = gelu if gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class TransformerModel(nn.Module):

    ATTRIBUTES = ['encoder', 'with_output', 'eos_index', 'pad_index', 'n_langs', 'n_words', 'dim', 'n_layers', 'n_heads', 'hidden_dim', 'dropout', 'attention_dropout', 'asm', 'asm_cutoffs', 'asm_div_value']

    # params : n_words, eos_index, pad_index, mask_token_id, emb_dim, dim_feedforward, n_heads, n_layers, dropout, attention_dropout, 
    # n_positions, gelu_activation, sinusoidal_embeddings, share_inout_emb, tim_layers_pos, n_s, use_group_comm
    # optional params : use_lang_emb (False) and n_langs (1) 

    def __init__(self, params, is_encoder, with_output):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        # encoder / decoder, output layer
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        # dictionary
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.use_lang_emb = getattr(params, 'use_lang_emb', False)
        self.n_langs = getattr(params, "n_langs", 1)

        # model parameters
        self.dim = params.emb_dim       # 512 by default
        #self.hidden_dim = self.dim * 4 # 2048 by default
        self.hidden_dim = params.dim_feedforward  # 2048 by default
        self.n_heads = params.n_heads   # 8 by default
        self.n_layers = params.n_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'

        # TIM layers positions an parameters
        self.tim_layers_pos = []
        if params.tim_layers_pos != "" : # and not self.is_decoder:
            self.tim_layers_pos = [int(pos) for pos in params.tim_layers_pos.split("-")]
            assert all([ 0 <= pos <= params.n_layers - 1 for pos in self.tim_layers_pos])
            self.n_s = params.n_s
           
        # embeddings
        self.position_embeddings = Embedding(params.n_positions, self.dim)
        if params.sinusoidal_embeddings:
            with torch.no_grad():
                create_sinusoidal_embeddings(params.n_positions, self.dim, out=self.position_embeddings.weight)
        if self.use_lang_emb and self.n_langs > 1 :
            self.lang_embeddings = Embedding(self.n_langs, self.dim)
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.tim_layers_pos :
            self.tim_layers = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            if layer_id in self.tim_layers_pos :
                if is_encoder :
                    self.tim_layers.append(
                        TransformerEncoderLayer(
                                                d_ffn=self.hidden_dim,
                                                nhead=self.n_heads,
                                                kdim=self.dim,
                                                vdim=self.dim,
                                                dropout=self.attention_dropout,
                                                activation= nn.GELU if params.gelu_activation else nn.ReLU,
                                                num_modules=self.n_s,
                                                use_group_comm = params.use_group_comm
                            )
                        )
                    _ = self.tim_layers[-1](src = torch.rand((1, 1, self.dim)), init_params = True)
                else :
                    self.tim_layers.append(
                        TransformerDecoderLayer(
                                                d_ffn=self.hidden_dim,
                                                nhead=self.n_heads,
                                                kdim=self.dim,
                                                vdim=self.dim,
                                                dropout=self.attention_dropout,
                                                activation= nn.GELU if params.gelu_activation else nn.ReLU,
                                                num_modules=self.n_s,
                                                use_group_comm = params.use_group_comm
                            ) 
                        )
                    _ = self.tim_layers[-1](tgt = torch.rand((1, 1, self.dim)), memory = torch.rand((1, 1, self.dim)), init_params = True)
            else :
                self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
                self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
                if self.is_decoder:
                    self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                    self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
                self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.dropout, gelu_activation=params.gelu_activation))
                self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        # output layer
        if self.with_output:
            self.pred_layer = PredLayer(params)
            if params.share_inout_emb:
                self.pred_layer.proj.weight = self.embeddings.weight

    def forward(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, langs=None,
                attention_mask = None, token_type_ids = None # these two parameters allow to have the same parameters as the hugging_face implementation
    ):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        #lengths = (x != self.pad_index).float().sum(dim=0)
        #mask = x != self.pad_index
        
        # check inputs
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.size() == (slen, bs)
            langs = langs.transpose(0, 1)

        # embeddings
        tensor = self.embeddings(x)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if langs is not None and self.use_lang_emb:
            tensor = tensor + self.lang_embeddings(langs)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        
        # transformer layers
        i, j = 0, 0
        for k in range(self.n_layers):
            if k in self.tim_layers_pos :
                src_mask = attn_mask.repeat(self.n_heads, 1, 1) if causal else None
                src_mask = ~src_mask.to(torch.bool) if src_mask is not None else None
                tensor,_ = self.tim_layers[j](tensor, src_mask = src_mask, src_key_padding_mask = ~mask.to(torch.bool))
                j += 1
            else :
                # self attention
                attn = self.attentions[i](tensor, attn_mask)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm1[i](tensor)
                # encoder attention (for decoder only)
                if self.is_decoder and src_enc is not None:
                    attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc)
                    attn = F.dropout(attn, p=self.dropout, training=self.training)
                    tensor = tensor + attn
                    tensor = self.layer_norm15[i](tensor)
                # FFN
                tensor = tensor + self.ffns[i](tensor)
                tensor = self.layer_norm2[i](tensor)
                i += 1
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor

    def predict(self, tensor, pred_mask, y):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
        """
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        scores, loss = self.pred_layer(masked_tensor, y)
        return scores, loss

    @staticmethod
    def generate(self, x, lengths, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None, langs = None):
        """
        Decode a sentence given initial start (src_enc and src_len for MT).
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
            - None (MT)
        `lengths`:
            - LongTensor(bs) 
            - None (MT)
        `src_enc` :
            - None (LM)
            - Tensor(bs, source_len, dim)
        `src_len`
            - None (LM)
            - LongTensor(bs) with src_len.max() == source_len
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """
        # TODO : sanity check
        assert (x is not None) ^ (src_enc is not None)
        if x is not None :
            x_len, bs = x.size()
            assert lengths.size(0) == bs
            assert max_len > x_len
            src_len = lengths
            bs = len(src_len)
        else : 
            x_len = 0
            bs = len(src_len)
            assert src_enc.size(0) == bs
        # generated sentences
        max_len = max_len + 1 #  <EOS> token
        generated = src_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)       # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)    # we use <EOS> for <BOS> everywhere
        if x is not None :    
            generated[1:x_len+1] = x
        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        # language IDs
        if langs is None :
            langs = src_len.new(max_len).long().fill_(tgt_lang_id)
            langs = langs.unsqueeze(1).expand(max_len, bs)
        else :
            assert langs.size() == (max_len, bs)
        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1 + x_len
        gen_len = src_len.clone().fill_(1) if x is None else lengths
        unfinished_sents = src_len.clone().fill_(1)
        src_len = None if src_enc is None else src_len
        while cur_len < max_len:
            # compute word scores
            tensor = self.forward(
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len 
            )
            #assert tensor.size() == (1, bs, self.dim), (cur_len, max_len, src_enc.size(), tensor.size(), (1, bs, self.dim))
            #assert tensor.size() == (1, bs, self.dim), (cur_len, max_len, x.size(), tensor.size(), (1, bs, self.dim))
            assert tensor.size() == (cur_len, bs, self.dim), (cur_len, max_len, tensor.size(), (1, bs, self.dim))
            tensor = tensor.data[-1, :, :]#.type_as(src_enc)  # (bs, dim) # TODO
            scores = self.pred_layer.get_scores(tensor)      # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            # to line is to solve :
            # /pytorch/aten/src/ATen/native/cuda/LegacyDefinitions.cpp:19: UserWarning: masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated,please use a mask with dtype torch.bool instead.
            #generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)

        # sanity check
        if self.eos_index == self.pad_index :
            assert (generated == self.eos_index).sum() == 2 * bs
        else :
            # TODO
            pass

        return generated[:cur_len], gen_len

    @staticmethod
    def generate_beam(self, x, lengths, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, 
                    early_stopping, max_len=200, langs = None):
        """
        Decode a sentence given initial start ((src_enc and src_len for MT)).
        `x`:
            - LongTensor(bs, slen) (LM)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
            - None (MT)
        `lengths`:
            - LongTensor(bs) 
            - None (MT)
        `src_enc` :
            - None (LM)
            - Tensor(bs, source_len, dim)
        `src_len`
            - None (LM)
            - LongTensor(bs) with src_len.max() == source_len
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # TODO : sanity check
        
        # check inputs
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_len)
        n_words = self.n_words

        # expand to beam size the source latent representations / source lengths
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(self.pad_index)                   # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)                # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)

        # language IDs
        if langs is None :
            langs = positions.clone().fill_(tgt_lang_id)
        else :
            assert langs.size() == positions.size()

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len
            )
            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[-1, :, :]               # (bs * beam_size, dim)
            scores = self.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)
            scores = F.log_softmax(scores, dim=-1)       # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)            # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
 
            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(tokenize.decode(x) for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        # sanity check
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty