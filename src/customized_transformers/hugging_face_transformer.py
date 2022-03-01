"""
code for vanilla Transformers (Attention is all you need) & BERT, using huggingface transformers library 
"""

import torch
import torch.nn as nn

from transformers import BertModel, BertConfig

from .vanilla_transformer import create_sinusoidal_embeddings, PredLayer, get_causal_masks, get_masks

class HFTransformer(nn.Module):
    """Customized Transformers for Causal and Mask Language Modeling"""
    def __init__(self, params, is_encoder, with_output):
        super().__init__()
        """
        params: n_words, eos_index, pad_index, mask_token_id, mlm_probability, emb_dim, dim_feedforward, n_heads, n_layers, dropout, 
                attention_dropout, 
                n_positions, gelu_activation, sinusoidal_embeddings, share_inout_emb, tim_layers_pos, n_s, use_group_comm
        optional params : use_lang_emb (False), n_langs (1), sample_alpha (0)
        """    
        config = BertConfig(
            vocab_size = params.n_words, hidden_size = params.emb_dim, num_hidden_layers = params.n_layers, 
            num_attention_heads = params.n_heads, intermediate_size = params.dim_feedforward, hidden_act = 'gelu' if params.gelu_activation else "relu", 
            hidden_dropout_prob = params.dropout, attention_probs_dropout_prob = params.attention_dropout, max_position_embeddings = params.n_positions, 
            type_vocab_size = 2, initializer_range = 0.02, layer_norm_eps = 1e-12, pad_token_id = params.pad_index, position_embedding_type = 'absolute', 
            use_cache = True, classifier_dropout = None, is_decoder = not is_encoder, add_cross_attention = not is_encoder)
        # https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L848
        self.bert = BertModel(config, add_pooling_layer=False)
        if params.sinusoidal_embeddings:
            with torch.no_grad(): # RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
                create_sinusoidal_embeddings(
                    config.max_position_embeddings, 
                    config.hidden_size, 
                    out = self.bert.embeddings.position_embeddings.weight
                )

        # output layer
        if with_output:
            #self.pred_layer = BertOnlyMLMHead(config)
            self.pred_layer = PredLayer(params)
            if params.share_inout_emb:
                self.pred_layer.proj.weight = self.bert.embeddings.word_embeddings.weight

        self.is_decoder = not is_encoder
        self.dim = params.emb_dim
        self.n_words = params.n_words

    # def forward(self, input_ids, attention_mask, token_type_ids, positions, causal : bool):
    #     """
    #     Inputs:
    #         `input_ids` LongTensor(bs, seq_len), containing word indices
    #         `labels` LongTensor(bs, seq_len)
    #         `attention_mask` FloatTensor(bs, seq_len)
    #         `token_type_ids` LongTensor(bs, seq_len)
    #         `causal` Boolean, if True, the attention is only done over previous hidden states (clm)
    #     """

    # allow to have the same parameters as our custom vanilla implementation
    def forward(self, x, lengths, causal, attention_mask, token_type_ids, src_enc=None, src_len=None, positions=None, langs=None):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            lengths, causal, src_en, src_len and langs allow to have the same parameters as the custom vanilla implementation
        
            - mask : Mask to avoid performing attention on the padding token indices of the encoder input. 
                This mask is used in the cross-attention if the model is configured as a decoder. 
                Mask values selected in [0, 1]
            - attn_mask : Mask to avoid performing attention on padding token indices. 
                Mask values selected in [0, 1]
            - 1 for tokens that are not masked and 0 for tokens that are masked.
        """

        # TODO : sanity check for causal mask
        causal_mask = None
        if causal :
            if True :
                mask, attn_mask = get_masks(x.size(0), lengths, causal = True) # (bs, x_len), (bs, x_len, x_len) 
                causal_mask, attention_mask = mask.int(), attn_mask.int()
            else :
                causal_mask = get_causal_masks(mask=attention_mask, slen=x.size(0))
                causal_mask = causal_mask.int()

        # if self.is_decoder and src_enc is not None:
        #     src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]
        
        outputs = self.bert(
            input_ids = x.transpose(0, 1), # (bs, slen)
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = positions,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=src_enc,
            encoder_attention_mask=causal_mask,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
        ) 
        #outputs.hidden_states
        #outputs.attentions
        tensor = outputs[0].transpose(0, 1) # (slen, bs, dim)
        return tensor

