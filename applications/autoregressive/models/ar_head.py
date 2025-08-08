import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def create_norm(norm_type: str, dim: int, eps: float = 1e-6):
    """
    Creates the specified normalization layer based on the norm_type.
    Adopted from TorchTriton: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/norms.py

    Args:
        norm_type (str): The type of normalization layer to create.
            Supported types: 1. rmsnorm 2. fused_rmsnorm 3. layernorm 4. np_layernorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        The created normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()  # Normalize to lowercase

    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps, compile=False)
    elif norm_type == "compiled_rmsnorm":
        return RMSNorm(dim, eps=eps, compile=True)
    elif norm_type == "fused_rmsnorm":
        raise NotImplementedError("Fused RMSNorm is not supported yet.")
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")

class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.
    Reference implementation: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/norms.py

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        compile (bool, optional): Whether to compile the forward function. Default is False.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6, compile: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.rmsnorm_fn = torch.compile(self.compute_rmsnorm, fullgraph=True) if compile else self.compute_rmsnorm

    @staticmethod
    def compute_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
        def _norm(x, eps):
            # Computes the root-mean-square norm of the input tensor.
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        output = _norm(x.float(), eps).type_as(x)
        return output * weight

    def forward(self, x: torch.Tensor):
        return self.rmsnorm_fn(x, self.weight, self.eps)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for k, v in kwargs.items():
            setattr(self, k, v)     

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
    mask: Optional[torch.Tensor] = None,
    is_causal: Optional[bool] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    PyTorch's native implementation of Flash Attention 2.

    If `is_causal` is given, then the causal attention mask is applied accordingly:
    - If `is_causal` is True, the standard upper-left causal attention masking is applied.
    - If `is_causal` is False, no attention mask is applied, unless an explicit mask tensor is
      provided (i.e., `mask is not None`).

    If `is_causal` is not given (i.e., `is_causal is None`), then the attention mask is applied
    based on the provided mask tensor:
    - If no explicit attention mask is given (i.e., `mask is None`), `is_causal` is set to True,
    leading to the standard upper-left causal attention masking.
    - If an attention mask is given (i.e., `mask is not None`), the provided mask is used,
    and `is_causal` is set to False.

    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        v (torch.Tensor): Value tensor
        head_dim (int): Dimension of each attention head
        mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
        is_causal (Optional[bool], optional): Whether to apply causal attention mask. Defaults to None.
        dropout_p (float, optional): Dropout rate. Defaults to 0.0.

    Returns:
        torch.Tensor: Output tensor after applying scaled dot-product attention
    """
    scale = 1.0 / math.sqrt(head_dim)
    if is_causal is None:
        is_causal = mask is None
    y = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=dropout_p,
        # scale=scale,
        is_causal=is_causal,
    )
    return y.transpose(1, 2).contiguous()

class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        """
        Initializes the multilayer perceptron (MLP) module.

        Args:
            dim: The input and output dimensionality.
            hidden_dim: The dimensionality of the hidden layer.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP module.

        Args:
            x: The input tensor of shape (batch_size, dim).

        Returns:
            The output tensor of shape (batch_size, dim).
        """
        output = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return output

class ImageCausalAttention(nn.Module):
    def __init__(
        self, 
        config, 
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        use_qk_normalization: bool = True,
        max_tokens_num = 1000
    ):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.qk_norm = use_qk_normalization
        self.q_norm = create_norm(norm_type, dim=config.n_embd  // self.n_head, eps=norm_eps)
        self.k_norm = create_norm(norm_type, dim=config.n_embd  // self.n_head, eps=norm_eps)

        self.max_tokens_num = max_tokens_num
        
    def forward(
        self, 
        x, 
        mask,
    ):
        B, seqlen, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        q = q.view(B, seqlen, self.n_head, C // self.n_head)
        k = k.view(B, seqlen, self.n_head, C // self.n_head)
        v = v.view(B, seqlen, self.n_head, C // self.n_head)
        
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        
        is_causal = None
        output = scaled_dot_product_attention(
            q,
            k,
            v,
            head_dim=self.n_head,
            mask=mask,
            is_causal=is_causal,
            dropout_p=0.0,
        )
        output = output.view(B, seqlen, -1)
        output = self.proj(output)
        return output

class ImageCausalBlock(nn.Module):
    def __init__(
        self, 
        config,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        max_tokens_num = 1000,
        num = 3
    ):
        super().__init__()
        self.attention_norm = create_norm(norm_type, config.n_embd, eps=norm_eps)
        self.ffn_norm = create_norm(norm_type, config.n_embd, eps=norm_eps)
        self.attention = ImageCausalAttention(config, max_tokens_num=max_tokens_num)
        self.feed_forward = MLP(
            dim=config.n_embd,
            hidden_dim=4 * config.n_embd,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(num * config.n_embd, 6 * config.n_embd, bias=False)
        )
        
    def forward(
        self, 
        x, 
        mask,
    ):
        h = x + self.attention(self.attention_norm(x), mask=mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class AR_Head(nn.Module):
    """
    Inherit from VQGAN
    Autoregress the next sub-token feature given the previous sub-token features

    """
    def __init__(self, vocab_size=2048, n_layer=2, n_embd=1536, 
                block_size=8, embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., 
                n_unmasked=0, n_head=4):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, n_head=n_head,block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop, 
                           n_unmasked=n_unmasked, n_layer=n_layer, n_embd=n_embd)
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.auto_regressive_num = config.n_layer
        self.causal_space_blocks = nn.Sequential(*[ImageCausalBlock(config) for _ in range(config.n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        self.n_head = n_head
        
        self.sos_emb_for_image = nn.Parameter(torch.zeros(1, 1, self.config.n_embd)) 
        nn.init.normal(self.sos_emb_for_image.data, mean=0, std=0.02)
        
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, feature, gt_index):
        """
        input:
            feature: B 1 C        
            ground truth index: B G
        output:
            logits: B G V
        """
        token_embeddings = self.tok_emb(gt_index)    # (B L) G C
        B, _, _ = token_embeddings.shape
        sos_emb_for_image = self.sos_emb_for_image.repeat((B, 1, 1)).contiguous()
        token_embeddings = torch.cat([feature, sos_emb_for_image, token_embeddings[:, :-1]], dim=1)    # B (1 + G) C

        t = token_embeddings.shape[1] # G+1
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)   # B (h*w + G) C #1+G?
        # x = self.causal_space_blocks(x, self.img_mask[:, :, :t, :t])
        for i in range(self.auto_regressive_num):
            x = self.causal_space_blocks[i](x, None)


        x = self.ln_f(x)    # B (h*w + G) C
        logits = self.head(x) # B (h*w + G) V

        return logits
    
    @torch.no_grad()
    def generate_gt_index(self, feature, gt_index = None):
        """
        input:
            feature: B 1 C        
            ground truth index: B G
        output:
            logits: B G V
        """
        B, _, _ = feature.shape
        sos_emb_for_image = self.sos_emb_for_image.repeat((B, 1, 1)).contiguous()
        if gt_index is not None:
            token_embeddings = self.tok_emb(gt_index)  
            token_embeddings = torch.cat([feature, sos_emb_for_image, token_embeddings], dim=1) 
        else:
            token_embeddings = torch.cat([feature, sos_emb_for_image], dim=1) 
            
        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        for i in range(self.auto_regressive_num):
            x = self.causal_space_blocks[i](x, None)
        x = self.ln_f(x)    # B X C 
        logits = self.head(x) # B X V

        return logits

