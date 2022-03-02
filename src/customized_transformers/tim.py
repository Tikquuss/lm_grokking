"""
code for Transformers with Independent Mechanisms & BERT 

Transformers with Competitive Ensembles of Independent Mechanisms
Alex Lamb, Di He, Anirudh Goyal, Guolin Ke, Chien-Feng Liao, Mirco Ravanelli, Yoshua Bengio
https://arxiv.org/abs/2103.00336
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class GroupLayerNorm(nn.Module):
    def __init__(
        self, dim, nb, eps=1e-6,
    ):
        super(GroupLayerNorm, self).__init__()

        self.num_rims = nb
        self.dim = dim
        self.eps = eps

    def init_params(self, first_input):
        self.dim = first_input.shape[-1]

        self.weight = nn.Parameter(
            torch.ones([1, 1, self.dim], device=first_input.device)
        )
        self.bias = nn.Parameter(
            torch.zeros([1, 1, self.dim], device=first_input.device)
        )

        self.norm = nn.LayerNorm(
            self.dim // self.num_rims, eps=self.eps, elementwise_affine=False
        ).to(first_input.device)

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_rims, self.dim // self.num_rims)

        x = self.norm(x)

        x = x.view(bsz, seq_len, self.dim)

        x = x * self.weight + self.bias

        return x

class GroupLinear(nn.Module):
    def __init__(self, din, dout, nb, bias=True, a=None):
        super(GroupLinear, self).__init__()
        self.nb = nb

        din = din // nb
        dout = dout // nb

        self.dout = dout

        if a is None:
            a = 1.0 / math.sqrt(dout)

        # gain = 1.0 / math.sqrt(2)
        # a = gain * math.sqrt(6.0 / (din + dout))

        self.weight = nn.Parameter(
            torch.FloatTensor(nb, din, dout).uniform_(-a, a)
        )

        self.bias = bias

        if bias is True:
            self.bias = nn.Parameter(
                torch.FloatTensor(nb, dout).uniform_(-a, a)
            )
            # self.bias = nn.Parameter(torch.zeros(dout*nb))
        else:
            self.bias = None

    def forward(self, x):

        # input: ts x bs x blocks*nhid
        # ts*bs , blocks, nhid
        # blocks, ts*bs, nhid
        bs, ts, m = x.shape

        x = x.reshape((bs * ts, self.nb, m // self.nb))
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.weight)
        x = x.permute(1, 0, 2)

        if self.bias is not None:
            x = x + self.bias

        x = x.reshape((bs, ts, self.dout * self.nb))

        # if not self.bias is None:
        #    x += self.bias

        return x

class GroupMLP(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, din, dout, nb, dropout=0.1):
        super(GroupMLP, self).__init__()

        self.w_1 = nn.Parameter(0.01 * torch.randn(nb, din, dout))
        self.w_2 = nn.Parameter(0.01 * torch.randn(nb, dout, din))

        self.layer_norm = nn.LayerNorm(din)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x * 1.0
        x = x.permute(1, 0, 2)
        x = torch.bmm(F.relu(torch.bmm(x, self.w_1)), self.w_2)
        x = x.permute(1, 0, 2)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x

class MechanismCommunication(nn.Module):
    def __init__(self, dim, n_blocks, n_heads = 2):
        super(MechanismCommunication, self).__init__()

        self.n_heads = n_heads
        #self.n_heads = 2
        self.n_blocks = n_blocks
        # self.head_dim = self.block_dim // self.n_heads
        self.head_dim = 32
        self.scale = self.head_dim ** -0.5

    def init_params(self, first_input):
        self.dim = first_input.shape[-1]
        self.block_dim = self.dim // self.n_blocks
        #####
        if True :
            self.head_dim = self.block_dim // self.n_heads
            self.scale = self.head_dim ** -0.5
        
        self.emb_dim = self.head_dim * self.n_heads * self.n_blocks

        self.query_net = GroupLinear(self.dim, self.emb_dim, self.n_blocks).to(
            first_input.device
        )
        self.key_net = GroupLinear(self.dim, self.emb_dim, self.n_blocks).to(
            first_input.device
        )
        self.value_net = GroupLinear(self.dim, self.emb_dim, self.n_blocks).to(
            first_input.device
        )
        self.final = GroupLinear(self.emb_dim, self.dim, self.n_blocks).to(
            first_input.device
        )

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        bsz, seq_len, _ = x.shape

        q = self.query_net(x).view(
            bsz, seq_len, self.n_blocks, self.n_heads, self.head_dim
        )
        k = self.key_net(x).view(
            bsz, seq_len, self.n_blocks, self.n_heads, self.head_dim
        )
        v = self.value_net(x).view(
            bsz, seq_len, self.n_blocks, self.n_heads, self.head_dim
        )

        q = q.transpose(2, 3) * self.scale
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        score = torch.matmul(q, k.transpose(3, 4))
        score = F.softmax(score, dim=-1)
        out = torch.matmul(score, v).transpose(2, 3)
        score = score.mean(dim=2)

        out = out.reshape(
            bsz, seq_len, self.n_blocks * self.head_dim * self.n_heads
        )
        out = self.final(out)
        out = out.view(bsz, seq_len, self.dim)

        return out

class PositionalEncoding(nn.Module):
    """This class implements the positional encoding function
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    Arguements
    ----------
    max_len :
        max length of the input sequences (default 2500)
    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding()
    >>> b = enc(a, init_params=True)
    >>> print(b.shape)
    torch.Size([1, 120, 512])
    """

    def __init__(self, max_len=2500):
        super().__init__()
        self.max_len = max_len

    def init_params(self, first_input):
        model_dim = first_input.shape[-1]
        pe = torch.zeros(self.max_len, model_dim, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, model_dim, 2).float()
            * -(math.log(10000.0) / model_dim)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0).to(first_input.device)
        self.register_buffer("pe", pe)

    def forward(self, x, init_params=False):
        """
        Arguements
        ----------
        x:
            input feature (batch, time, fea)
        """
        if init_params:
            self.init_params(x)
        return self.pe[:, : x.size(1)].clone().detach()

class PositionalwiseFeedForward(nn.Module):
    def __init__(self, d_ffn, nb=1, dropout=0.1, activation=nn.ReLU):
        """The class implements the positional-wise feadd forward module in “Attention Is All You Need”

        Arguements
        ----------
        d_ffn: int
            dimention of representation space of this positional-wise feadd forward module
        dropout: float
            dropout
        activation: torch class
            activation functions to be applied (Recommandation: ReLU, GELU)
        """
        super().__init__()
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.activation = activation
        self.nb = nb

    def init_params(self, first_input):
        self.input_size = first_input.shape[-1]

        self.ffn = nn.Sequential(
            GroupLinear(self.input_size, self.d_ffn, nb=self.nb),
            self.activation(),
            nn.Dropout(self.dropout),
            GroupLinear(self.d_ffn, self.input_size, nb=self.nb),
        ).to(first_input.device)

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        # give a tensor of shap (time, batch, fea)
        x = x.permute(1, 0, 2)

        x = self.ffn(x)

        # reshape the output back to (batch, time, fea)
        x = x.permute(1, 0, 2)

        return x

class MultiheadAttention(nn.Module):
    """ The class is a wrapper of MultiHead Attention for torch.nn.MultiHeadAttention.
    ref: https://pytorch.org/docs/stable/nn.html

    Arguements
    ----------
    num_heads : int
        parallel attention heads.
    dropout : float
        a Dropout layer on attn_output_weights. Default: 0.0.
    bias : bool
        add bias as module parameter. Default: True.
    add_bias_kv : bool
        add bias to the key and value sequences at dim=0.
    add_zero_attn : bool
        add a new batch of zeros to the key and value sequences at dim=1.
    kdim : int
        total number of features in key. Default: None.
    vdim : int
        total number of features in value. Default: None.
    """

    def __init__(
        self,
        nhead,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        nb=1,
    ):
        super().__init__()
        self.nhead = nhead
        self.dropout = dropout
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.kdim = kdim
        self.vdim = vdim
        self.nb = nb

    def init_params(self, first_input):
        if len(first_input.shape) == 4:
            first_input = first_input.reshape(
                first_input.shape[0],
                first_input.shape[1],
                first_input.shape[2] * first_input.shape[3],
            )

        self.embed_dim = first_input.shape[-1] // self.nb

        if self.kdim is not None:
            self.kdim = self.kdim // self.nb

        if self.vdim is not None:
            self.vdim = self.vdim // self.nb

        self.att = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.nhead // self.nb,
            dropout=self.dropout,
            bias=self.bias,
            add_bias_kv=self.add_bias_kv,
            add_zero_attn=self.add_zero_attn,
            kdim=self.kdim,
            vdim=self.vdim,
        ).to(first_input.device)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        init_params=False,
    ):
        """
        Arguements
        ----------
        query: tensor
            (L, N, E)(L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
        key: tensor
            (S, N, E)(S,N,E) , where S is the source sequence length, N is the batch size, E is the embedding dimension.
        value: tensor
            (S, N, E)(S,N,E) where S is the source sequence length, N is the batch size, E is the embedding dimension.
        key_padding_mask: tensor
            (N, S)(N,S) where N is the batch size, S is the source sequence length. If a ByteTensor is provided, the non-zero positions will be ignored while the position with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the value of True will be ignored while the position with the value of False will be unchanged.
        attn_mask: tensor
            2D mask (L, S)(L,S) where L is the target sequence length, S is the source sequence length. 3D mask (N*num_heads, L, S)(N∗num_heads,L,S) where N is the batch size, L is the target sequence length, S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged. If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged. If a FloatTensor is provided, it will be added to the attention weight.

        Outputs
        -------
        attn_output: tensor
            (L, N, E)(L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
        attn_output_weights: tensor
            (N, L, S)(N,L,S) where N is the batch size, L is the target sequence length, S is the source sequence length.
        """
        if init_params:
            self.init_params(key)

        # give tensors of shape (time, batch, fea)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        tq, bsz, _ = query.shape

        query = query.reshape(
            (
                query.shape[0],
                query.shape[1] * self.nb,
                query.shape[2] // self.nb,
            )
        )
        key = key.reshape(
            (key.shape[0], key.shape[1] * self.nb, key.shape[2] // self.nb)
        )
        value = value.reshape(
            (
                value.shape[0],
                value.shape[1] * self.nb,
                value.shape[2] // self.nb,
            )
        )

        if key_padding_mask is not None:
            key_padding_mask = (
                key_padding_mask.unsqueeze(1)
                .repeat(1, self.nb, 1)
                .reshape(
                    (
                        key_padding_mask.shape[0] * self.nb,
                        key_padding_mask.shape[1],
                    )
                )
            )

        output, attention = self.att(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        output = output.reshape((tq, bsz, output.shape[2] * self.nb))

        # reshape the output back to (batch, time, fea)
        output = output.permute(1, 0, 2)

        return output, attention

class TransformerEncoderLayer(nn.Module):
    """ This is an implementation of self-attention encoder layer
    Arguements
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer
    nhead : int
        number of attention heads
    kdim : int
        dimension for key (Optional)
    vdim : int
        dimension for value (Optional)
    dropout : int
        dropout for the encoder (Optional)
    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoderLayer(512, 8)
    >>> output = net(x, init_params=True)
    >>> print(output[0].shape)
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
        num_modules=1,
        use_group_comm=False,
    ):
        super().__init__()
        self.self_att = MultiheadAttention(
            nhead=nhead, dropout=dropout, kdim=kdim, vdim=vdim, nb=num_modules,
        )
        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn, dropout=dropout, activation=activation, nb=num_modules,
        )

        self.num_modules = num_modules
        self.d_ffn = d_ffn

        self.norm1 = GroupLayerNorm(d_ffn, num_modules, eps=1e-6)
        self.norm2 = GroupLayerNorm(d_ffn, num_modules, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.use_group_comm = use_group_comm
        if use_group_comm:
            self.group_comm = MechanismCommunication(d_ffn, num_modules)
            self.norm_comm = GroupLayerNorm(d_ffn, num_modules, eps=1e-6)
            self.dropout_comm = torch.nn.Dropout(dropout)

    def init_params(self, first_input):
        self.din = first_input.shape[-1]

        if self.num_modules > 1:
            self.competition = GroupLinear(
                self.din, self.num_modules, self.num_modules, a=0.05
            ).to(first_input.device)
        else:
            self.competition = None

    def forward(
        self, src, src_mask=None, src_key_padding_mask=None, init_params=False
    ):
        """
        Arguements
        ----------
        src: tensor
            the sequence to the encoder layer (required).
        src_mask: tensor
            the mask for the src sequence (optional).
        src_key_padding_mask: tensor
            the mask for the src keys per batch (optional).
        """
        if init_params:
            self.init_params(src)

        if self.competition is not None:
            comp = self.competition(src)
            comp = F.softmax(comp, dim=2)
            self.comp_score = comp
            comp = comp.unsqueeze(-1).repeat(
                1, 1, 1, self.din // self.num_modules
            )
            comp = comp.view((src.shape[0], src.shape[1], self.din))
        else:
            comp = 1.0
            self.comp_score = None

        output, self_attn = self.self_att(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            init_params=init_params,
        )

        # add & norm
        src = src + self.dropout1(output) * comp
        src = self.norm1(src, init_params=init_params)

        output = self.pos_ffn(src, init_params)

        # add & norm
        output = src + self.dropout2(output)
        output = self.norm2(output, init_params=init_params)

        if self.use_group_comm:
            residual = output * 1.0
            output = self.group_comm(output, init_params=init_params)
            output = self.dropout_comm(output)
            output = self.norm_comm(output + residual, init_params=init_params)

        return output, self.comp_score

class TransformerDecoderLayer(nn.Module):
    """This class implements the self-attention decoder layer
    Arguements
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer
    nhead : int
        number of attention heads
    kdim : int
        dimension for key (optional)
    vdim : int
        dimension for value (optional)
    dropout : float
        dropout for the decoder (optional)
    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = TransformerDecoderLayer(1024, 8)
    >>> output = net(src, tgt, init_params=True)
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
        num_modules=1,
        use_group_comm=False,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(
            nhead=nhead, kdim=kdim, vdim=vdim, dropout=dropout, nb=num_modules,
        )
        self.mutihead_attn = MultiheadAttention(
            nhead=nhead, kdim=kdim, vdim=vdim, dropout=dropout, nb=num_modules,
        )
        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn, dropout=dropout, activation=activation, nb=num_modules,
        )

        self.num_modules = num_modules
        self.d_ffn = d_ffn
        if num_modules > 1:
            self.competition = GroupLinear(
                d_ffn // num_modules, 1, num_modules, a=0.05
            )
        else:
            self.competition = None

        self.use_group_comm = use_group_comm
        if use_group_comm:
            self.group_comm = MechanismCommunication(d_ffn, num_modules)
            self.norm_comm = GroupLayerNorm(d_ffn, num_modules, eps=1e-6)
            self.dropout_comm = torch.nn.Dropout(dropout)

        # normalization layers
        self.norm1 = GroupLayerNorm(d_ffn, num_modules, eps=1e-6)
        self.norm2 = GroupLayerNorm(d_ffn, num_modules, eps=1e-6)
        self.norm3 = GroupLayerNorm(d_ffn, num_modules, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        init_params=False,
    ):
        """
        Arguements
        ----------
        tgt: tensor
            the sequence to the decoder layer (required).
        memory: tensor
            the sequence from the last layer of the encoder (required).
        tgt_mask: tensor
            the mask for the tgt sequence (optional).
        memory_mask: tensor
            the mask for the memory sequence (optional).
        tgt_key_padding_mask: tensor
            the mask for the tgt keys per batch (optional).
        memory_key_padding_mask: tensor
            the mask for the memory keys per batch (optional).
        """

        if self.competition is not None:
            comp = self.competition(tgt)
            comp = F.softmax(comp, dim=2)
            comp = comp.unsqueeze(-1).repeat(
                1, 1, 1, self.d_ffn // self.num_modules
            )
            comp = comp.view((tgt.shape[0], tgt.shape[1], self.d_ffn))
        else:
            comp = 1.0

        # self-attention over the target sequence
        tgt2, self_attn = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            init_params=init_params,
        )

        # add & norm
        tgt = tgt + self.dropout1(tgt2) * comp
        tgt = self.norm1(tgt, init_params)

        # multi-head attention over the target sequence and encoder states
        tgt2, multihead_attention = self.mutihead_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            init_params=init_params,
        )

        # add & norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt, init_params)

        tgt = self.pos_ffn(tgt, init_params)

        # add & norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt, init_params)

        if self.use_group_comm:
            residual = tgt * 1.0
            tgt = self.group_comm(tgt, init_params=init_params)
            tgt = self.dropout_comm(tgt)
            tgt = self.norm_comm(tgt + residual, init_params=init_params)

        return tgt, self_attn, multihead_attention

class TransformerEncoder(nn.Module):
    """This class implements the transformer encoder
    Arguements
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer
    nhead : int
        number of attention heads
    kdim : int
        dimension for key (Optional)
    vdim : int
        dimension for value (Optional)
    dropout : float
        dropout for the encoder (Optional)
    input_module: torch class
        the module to process the source input feature to expected feature dimension (Optional)
    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, 512)
    >>> output = net(x, init_params=True)
    >>> print(output.shape)
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
        return_attention=False,
        num_modules=1,
        use_group_comm=False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    num_modules=num_modules
                    if (j > 1 and j < num_layers - 1)
                    else 1,
                    use_group_comm=use_group_comm,
                )
                for j in range(num_layers)
            ]
        )
        self.norm = GroupLayerNorm(d_ffn, 1, eps=1e-6)
        self.return_attention = return_attention

    def forward(
        self, src, src_mask=None, src_key_padding_mask=None, init_params=False
    ):
        """
        Arguements
        ----------
        src: tensor
            the sequence to the encoder layer (required).
        src_mask: tensor
            the mask for the src sequence (optional).
        src_key_padding_mask: tensor
            the mask for the src keys per batch (optional).
        """
        output = src
        attention_lst = []
        for enc_layer in self.layers:
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                init_params=init_params,
            )
            attention_lst.append(attention)
        output = self.norm(output, init_params=init_params)

        if self.return_attention:
            return output, attention_lst
        return output

class TransformerDecoder(nn.Module):
    """This class implements the Transformer decoder
    Arguements
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer
    nhead : int
        number of attention heads
    kdim : int
        dimension for key (Optional)
    vdim : int
        dimension for value (Optional)
    dropout : float
        dropout for the decoder (Optional)
    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = TransformerDecoder(1, 8, 1024)
    >>> output = net(src, tgt, init_params=True)
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
        return_attention=False,
        num_modules=1,
        use_group_comm=False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    num_modules=num_modules
                    if (j > 1 and j < num_layers - 1)
                    else 1,
                    use_group_comm=use_group_comm,
                )
                for j in range(num_layers)
            ]
        )
        self.norm = GroupLayerNorm(d_ffn, 1, eps=1e-6)
        self.return_attention = return_attention

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        init_params=False,
    ):
        """
        Arguements
        ----------
        tgt: tensor
            the sequence to the decoder layer (required).
        memory: tensor
            the sequence from the last layer of the encoder (required).
        tgt_mask: tensor
            the mask for the tgt sequence (optional).
        memory_mask: tensor
            the mask for the memory sequence (optional).
        tgt_key_padding_mask: tensor
            the mask for the tgt keys per batch (optional).
        memory_key_padding_mask: tensor
            the mask for the memory keys per batch (optional).
        """
        output = tgt
        self_attns, multihead_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                init_params=init_params,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
        output = self.norm(output, init_params=init_params)

        if self.return_attention:
            return output, self_attns, multihead_attns
        return output


class TransformerInterface(nn.Module):
    """This is an interface for transformer model. Users can modify the attributes and
    define the forward function as needed according to their own tasks.
    The architecture is based on the paper "Attention Is All You Need": https://arxiv.org/pdf/1706.03762.pdf
    Arguements
    ----------
    d_model: int
        the number of expected features in the encoder/decoder inputs (default=512).
    nhead: int
        the number of heads in the multiheadattention models (default=8).
    num_encoder_layers: int
        the number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers: int
        the number of sub-decoder-layers in the decoder (default=6).
    dim_ffn: int
        the dimension of the feedforward network model (default=2048).
    dropout: int
        the dropout value (default=0.1).
    activation: torch class
        the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu)
    custom_src_module: torch class
        module that process the src features to expected feature dim
    custom_tgt_module: torch class
        module that process the src features to expected feature dim
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        custom_src_module=None,
        custom_tgt_module=None,
        return_attention=False,
        positional_encoding=True,
        num_modules=1,
        use_group_comm=False,
    ):
        super().__init__()

        assert (
            num_encoder_layers + num_decoder_layers > 0
        ), "number of encoder layers and number of decoder layers cannot both be 0!"

        if positional_encoding:
            self.positional_encoding = PositionalEncoding()

        # initialize the encoder
        if num_encoder_layers > 0:
            if custom_src_module is not None:
                self.custom_src_module = custom_src_module(d_model)

            self.encoder = TransformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                return_attention=return_attention,
                num_modules=num_modules,
                use_group_comm=use_group_comm,
            )

        # initialize the dncoder
        if num_encoder_layers > 0:
            if custom_tgt_module is not None:
                self.custom_tgt_module = custom_tgt_module(d_model)

            self.decoder = TransformerDecoder(
                num_layers=num_decoder_layers,
                nhead=nhead,
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                return_attention=return_attention,
                num_modules=num_modules,
                use_group_comm=use_group_comm,
            )

    def forward(self, **kwags):
        """Users should modify this function according to their own tasks
        """
        raise NotImplementedError


def get_key_padding_mask(padded_input, pad_idx):
    """Create a binary mask to prevent attention to padded locations
    Arguements
    ----------
    padded_input: int
        padded input
    pad_idx:
        idx for padding element
    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> km = get_key_padding_mask(a, pad_idx=0)
    >>> print(km)
    tensor([[False, False,  True],
            [False, False,  True],
            [False, False,  True]])
    """
    if len(padded_input.shape) == 4:
        bz, time, ch1, ch2 = padded_input.shape
        padded_input = padded_input.reshape(bz, time, ch1 * ch2)

    key_padded_mask = padded_input.eq(pad_idx)

    # if the input is more than 2d, mask the locations where they are silence across all channels
    if len(padded_input.shape) > 2:
        key_padded_mask = key_padded_mask.float().prod(dim=-1).bool()
        return key_padded_mask.detach()

    return key_padded_mask.detach()


def get_lookahead_mask(padded_input):
    """Creates a binary mask for each sequence.
    Arguements
    ----------
    padded_input : tensor
    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> sm = get_lookahead_mask(a)
    >>> print(sm)
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    seq_len = padded_input.shape[1]
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.detach().to(padded_input.device)