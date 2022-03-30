import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .vanilla_transformer import TransformerModel
from .hugging_face_transformer import HFTransformer

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Custom2HuggingFace(nn.Module):
    word_mask_keep_rand = "0.8,0.1,0.1" # Fraction of words to mask out / keep / randomize, among the words to predict
    # Like https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/data/data_collator.py#L769-L778
    def __init__(self, params, task):
        """
        params: hf_transformer, n_words, eos_index, pad_index, mask_token_id, mlm_probability, emb_dim, dim_feedforward, n_heads, n_layers, dropout, attention_dropout, 
                n_positions, gelu_activation, sinusoidal_embeddings, share_inout_emb, tim_layers_pos, n_s, use_group_comm
        optional params : use_lang_emb (False), n_langs (1), sample_alpha (0)
        """
        super().__init__()
        self.hugging_face_transformer = params.get("hf_transformer", False)
        self.pad_index = params.pad_index
        self.task = task
        self.causal = task == 'clm'

        if not self.hugging_face_transformer :
            self.transformer = TransformerModel(params, is_encoder=True, with_output=True)
        else :
            self.transformer = HFTransformer(params, is_encoder=True, with_output=True)
            setattr(self.transformer, "generate", TransformerModel.generate)
            setattr(self.transformer, "generate_beam", TransformerModel.generate_beam)

        self.n_words = params.n_words
        if self.causal :
            self.context_size = params.context_size
        else :
            #self.n_words = params.n_words
            self.mask_index = params.mask_token_id
            self.sample_alpha = getattr(params, 'sample_alpha', 0) # Exponent for transforming word counts to probabilities (~word2vec sampling)
            self.word_pred = params.mlm_probability
            # probability of masking out / randomize / not modify words to predict
            word_mask, word_keep, word_rand = [float(p) for p in self.word_mask_keep_rand.split(",")]
            self.pred_probs = torch.FloatTensor([word_mask, word_keep, word_rand])
            # TODO : load data_counts from data 
            # The value for a given token must correspond to the number of times this token appears in the dataset, not 1 as here
            # The idea is to make sure that the most frequent words are more likely to be masked than other words
            data_counts = {k: 1 for k in range(self.n_words)}
            counts = np.array(list(data_counts.values()))
            # TODO : make sure counts is 0 for special symbols (CLS_WORD, PAD_WORD, SEP_WORD, UNK_WORD, MASK_WORD, BOS_WORD, EOS_WORD ...)
            # probabilty to predict a word 
            params.mask_scores = np.maximum(counts, 1) ** -self.sample_alpha
            params.mask_scores[self.pad_index] = 0  # do not predict <PAD> index
            params.mask_scores[counts == 0] = 0     # do not predict special tokens

    def fwd(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_hidden_states=False,
    ):
        """
        if not self.hugging_face_transformer :
            x = input_ids.transpose(0, 1) # (bs, seq_len) -> (seq_len, bs)
            #lengths = (input_ids != self.pad_index).float().sum(dim=1).to(input_ids.device)
            lengths = (x != self.pad_index).float().sum(dim=0).to(input_ids.device)
            tensor = self.transformer(x = x, lengths = lengths, causal = self.causal, positions = position_ids) # (seq_len, bs, dim)
        else :
            output = self.transformer(input_ids, attention_mask, token_type_ids, position_ids, causal = self.causal) # (bs, seq_len, dim)
            #outputs.hidden_states
            #outputs.attentions,
            tensor = output[0].transpose(0, 1) # (seq_len, bs, dim)
        """  
        x = input_ids.transpose(0, 1) # (bs, seq_len) -> (seq_len, bs)
        #lengths = (input_ids != self.pad_index).long().sum(dim=1).to(input_ids.device)
        lengths = (x != self.pad_index).long().sum(dim=0).to(input_ids.device)
        tensor, hidden_states = self.transformer(x = x, lengths = lengths, causal = self.causal, positions = position_ids,
                attention_mask = attention_mask, token_type_ids = token_type_ids, output_hidden_states = output_hidden_states
        ) # (seq_len, bs, dim)

        return tensor, x, lengths, hidden_states

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        """All the options (and sub-options) below work normally"""
        flag = True 
        if flag :
            hidden_states, _, _, all_hidden_states = self.fwd(input_ids, attention_mask, token_type_ids, position_ids, output_hidden_states)
            logits = self.transformer.pred_layer.proj(hidden_states.transpose(0, 1)) # (bs, seq_len, vocab_size))
            if labels is not None :
                loss_fct = torch.nn.CrossEntropyLoss()
                if self.task == 'clm' :
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                elif self.task == 'mlm':
                    # -100 index = padding token
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        else :
            hidden_states, x, lengths, all_hidden_states = self.fwd(input_ids, attention_mask, token_type_ids, position_ids, output_hidden_states)
            if self.task == 'clm' :
                # Shift so that tokens < n predict n
                alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
                pred_mask = alen[:, None] < lengths[None] - 1
                if self.context_size > 0:  # do not predict without context
                    pred_mask[:self.context_size] = 0
                y = x[1:].masked_select(pred_mask[:-1])
                assert pred_mask.sum().item() == y.size(0)
            elif self.task == 'mlm':
                # https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/data/data_collator.py#L767
                if True :
                    # -100 index = padding token
                    pred_mask = labels != -100
                    y = labels[pred_mask]
                    pred_mask = pred_mask.transpose(0, 1) # (seq_len, bs)
                else :
                    # TOAVOID : params.mask_score has been incorrectly initialized (also, this method needs to call forward a second time to work)
                    x, y, pred_mask = self.mask_out(x)
                    hidden_states, x, lengths, all_hidden_states = self.fwd(x.transpose(0, 1), attention_mask, token_type_ids, position_ids)
            
            if False :
                # TOAVOID : it reduces the dimensions of the logits, which will produce an error when computing the accuracy
                logits, loss = self.transformer.predict(hidden_states, pred_mask, y)
            else :
                logits = self.transformer.pred_layer.proj(hidden_states.transpose(0, 1)) # (bs, seq_len, vocab_size))
                pred_mask = pred_mask.transpose(0, 1) # (bs, seq_len)
                masked_tensor = logits[pred_mask.unsqueeze(-1).expand_as(logits)].view(-1, self.n_words)
                loss = F.cross_entropy(masked_tensor, y)
            
        if not return_dict:
            return logits, loss

        return AttrDict({"loss" : loss, "logits" : logits, "hidden_states" : all_hidden_states, "attentions" : None})
    
    def mask_out(self, x, fp16 = False):
        """
        Decide of random words to mask out, and what target they get assigned.
        """            
        slen, bs = x.size()        

        # define target words to predict
        if self.sample_alpha == 0:
            pred_mask = np.random.rand(slen, bs) <= self.word_pred
            pred_mask = torch.from_numpy(pred_mask.astype(np.uint8))
        else:
            x_prob = self.mask_scores[x.flatten()]
            n_tgt = math.ceil(self.word_pred * slen * bs)
            tgt_ids = np.random.choice(len(x_prob), n_tgt, replace=False, p=x_prob / x_prob.sum())
            pred_mask = torch.zeros(slen * bs, dtype=torch.uint8)
            pred_mask[tgt_ids] = 1
            pred_mask = pred_mask.view(slen, bs)

        # do not predict padding
        pred_mask[x == self.pad_index] = 0
        pred_mask[0] = 0  # TODO: remove

        # mask a number of words == 0 [8] (faster with fp16)
        if fp16:
            pred_mask = pred_mask.view(-1)
            n1 = pred_mask.sum().item()
            n2 = max(n1 % 8, 8 * (n1 // 8))
            if n2 != n1:
                pred_mask[torch.nonzero(pred_mask).view(-1)[:n1 - n2]] = 0
            pred_mask = pred_mask.view(slen, bs)
            assert pred_mask.sum().item() % 8 == 0

        # generate possible targets / update x input
        pred_mask = pred_mask.bool() 
        _x_real = x[pred_mask]
        _x_rand = _x_real.clone().random_(self.n_words)
        _x_mask = _x_real.clone().fill_(self.mask_index)
        probs = torch.multinomial(self.pred_probs, len(_x_real), replacement=True)
        _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
        x = x.masked_scatter(pred_mask, _x)

        assert 0 <= x.min() <= x.max() < self.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    # def generate(self, input_ids, max_length, num_beams=1, early_stopping=False, 
    #             no_repeat_ngram_size=1, do_sample=False, top_k=0, top_p=0, num_return_sequences=1,
    #             decoder_start_token_id = None
    # ):
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor = None, #: Optional[LogitsProcessorList] = LogitsProcessorList(),
        stopping_criteria = None, #: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        """https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/generation_utils.py#L801"""
        x = inputs.transpose(0, 1) # (bs, seq_len) -> (seq_len, bs)
        lengths = (x != self.pad_index).long().sum(dim=0).to(inputs.device)
        if num_beams is None or num_beams == 1:
            generated, lengths = TransformerModel.generate(
                self.transformer,
                x=x, 
                lengths=lengths, 
                src_enc=None, 
                src_len=None, 
                tgt_lang_id=0, 
                max_len=max_length, 
                sample_temperature=temperature, 
                langs=None
            )
        else:
            length_penalty = 1.0 # values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.
            generated, lengths = TransformerModel.generate_beam(
                self.transformer,
                x=x, 
                lengths=lengths, 
                src_enc = None, 
                src_len = None, 
                tgt_lang_id = 0, 
                beam_size=num_beams,
                length_penalty=length_penalty, 
                early_stopping=early_stopping, 
                max_len=max_length, 
                langs=None
            )

        return generated

if __name__ == "__main__":

    vocab_size = 10
    max_length = 10

    params = AttrDict({
        "n_words":10, 
        "eos_index" : 0, 
        "pad_index" : 1, 
        "emb_dim" : 20, 
        "dim_feedforward" : 32, 
        "n_heads" : 4, 
        "n_layers" : 6, 
        "dropout" : 0.1, 
        "attention_dropout" : 0.1, 
        "n_positions" : max_length, 
        "gelu_activation" : True,
        "sinusoidal_embeddings" : True, 
        "share_inout_emb" : True,
        "tim_layers_pos":"2-3-4",
        "n_s" : 2,
        "use_group_comm" : True,

        "use_lang_emb" : False, 
        "n_langs" : 2
    })

    # model = HFTransformer(params, is_encoder=True, with_output=True)
    # setattr(model, "generate", TransformerModel.generate)
    # setattr(model, "generate_beam", TransformerModel.generate_beam)

    model = TransformerModel(params, is_encoder=True, with_output=True)
    #model.load_state_dict("../../../checkpoints.pth")

    slen, bs = 5, 2
    x = torch.randint(high=vocab_size, size =(slen, bs),  dtype=torch.long) 
    lengths = torch.randint(high=slen+1, size = (bs,),  dtype=torch.long) 
    
    h = model(x, lengths, causal=False, src_enc=None, src_len=None, positions=None, langs=None)
    print(h)
    
    # generate text - translate / convert to text
    max_len = int(1.5 * lengths.max().item() + 10)
    generated, lengths = TransformerModel.generate(
        model,
        x=x, 
        lengths=lengths, 
        src_enc=None, 
        src_len=None, 
        tgt_lang_id=0, 
        max_len=max_length, 
        sample_temperature=None, 
        langs=None
    )

    print(generated)