from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import MISSING, OmegaConf
import torch.nn.functional as F

from efficientvit.models.nn.act import build_act
from efficientvit.models.nn.norm import build_norm
from efficientvit.models.nn.ops import (
    ChannelDuplicatingPixelUnshuffleUpSampleLayer,
    ConvLayer,
    ConvPixelShuffleUpSampleLayer,
    ConvPixelUnshuffleDownSampleLayer,
    EfficientViTBlock,
    IdentityLayer,
    InterpolateConvUpSampleLayer,
    OpSequential,
    PixelUnshuffleChannelAveragingDownSampleLayer,
    ResBlock,
    ResidualBlock,
)

__all__ = ["MGVQ", "mgvq_f8c32","mgvq_f16c32", "mgvq_f32c32"]


@dataclass
class EncoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: str = "trms2d"
    act: str = "silu"
    downsample_block_type: str = "ConvPixelUnshuffle"
    downsample_match_channel: bool = True
    downsample_shortcut: Optional[str] = "averaging"
    out_norm: Optional[str] = None
    out_act: Optional[str] = None
    out_shortcut: Optional[str] = "averaging"
    double_latent: bool = False


@dataclass
class DecoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING
    in_shortcut: Optional[str] = "duplicating"
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: Any = "trms2d"
    act: Any = "silu"
    upsample_block_type: str = "ConvPixelShuffle"
    upsample_match_channel: bool = True
    upsample_shortcut: str = "duplicating"
    out_norm: str = "trms2d"
    out_act: str = "relu"


@dataclass
class MGVQConfig:
    in_channels: int = 3
    latent_channels: int = 32
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(in_channels="${..in_channels}", latent_channels="${..latent_channels}")
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderConfig(in_channels="${..in_channels}", latent_channels="${..latent_channels}")
    )
    use_quant_conv: bool = False
    codebook_size: int = 0
    codebook_groups: int = 0

    pretrained_path: Optional[str] = None
    pretrained_source: str = "mgvq"

    scaling_factor: Optional[float] = None


def build_block(
    block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str]
) -> nn.Module:
    if block_type == "ResBlock":
        assert in_channels == out_channels
        main_block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
        )
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_type == "EViT_GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=())
    elif block_type == "EViTS5_GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(5,))
    else:
        raise ValueError(f"block_type {block_type} is not supported")
    return block


def build_stage_main(
    width: int, depth: int, block_type: str | list[str], norm: str, act: str, input_width: int
) -> list[nn.Module]:
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            in_channels=width if d > 0 else input_width,
            out_channels=width,
            norm=norm,
            act=act,
        )
        stage.append(block)
    return stage


def build_downsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "Conv":
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_bias=True,
            norm=None,
            act_func=None,
        )
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


def build_upsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_encoder_project_in_block(in_channels: int, out_channels: int, factor: int, downsample_block_type: str):
    if factor == 1:
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
        )
    elif factor == 2:
        block = build_downsample_block(
            block_type=downsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
        )
    else:
        raise ValueError(f"downsample factor {factor} is not supported for encoder project in")
    return block


def build_encoder_project_out_block(
    in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str], shortcut: Optional[str]
):
    block = OpSequential(
        [
            build_norm(norm),
            build_act(act),
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
            ),
        ]
    )
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for encoder project out")
    return block


def build_decoder_project_in_block(in_channels: int, out_channels: int, shortcut: Optional[str]):
    block = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        use_bias=True,
        norm=None,
        act_func=None,
    )
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for decoder project in")
    return block


def build_decoder_project_out_block(
    in_channels: int, out_channels: int, factor: int, upsample_block_type: str, norm: Optional[str], act: Optional[str]
):
    layers: list[nn.Module] = [
        build_norm(norm, in_channels),
        build_act(act),
    ]
    if factor == 1:
        layers.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
            )
        )
    elif factor == 2:
        layers.append(
            build_upsample_block(
                block_type=upsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
            )
        )
    else:
        raise ValueError(f"upsample factor {factor} is not supported for decoder project out")
    return OpSequential(layers)


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )

        self.project_in = build_encoder_project_in_block(
            in_channels=cfg.in_channels,
            out_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            factor=1 if cfg.depth_list[0] > 0 else 2,
            downsample_block_type=cfg.downsample_block_type,
        )

        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            stage = build_stage_main(
                width=width, depth=depth, block_type=block_type, norm=cfg.norm, act=cfg.act, input_width=width
            )

            if stage_id < num_stages - 1 and depth > 0:
                downsample_block = build_downsample_block(
                    block_type=cfg.downsample_block_type,
                    in_channels=width,
                    out_channels=cfg.width_list[stage_id + 1] if cfg.downsample_match_channel else width,
                    shortcut=cfg.downsample_shortcut,
                )
                stage.append(downsample_block)
            self.stages.append(OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_encoder_project_out_block(
            in_channels=cfg.width_list[-1],
            out_channels=2 * cfg.latent_channels if cfg.double_latent else cfg.latent_channels,
            norm=cfg.out_norm,
            act=cfg.out_act,
            shortcut=cfg.out_shortcut,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        assert isinstance(cfg.act, str) or (isinstance(cfg.act, list) and len(cfg.act) == num_stages)

        self.project_in = build_decoder_project_in_block(
            in_channels=cfg.latent_channels,
            out_channels=cfg.width_list[-1],
            shortcut=cfg.in_shortcut,
        )

        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                upsample_block = build_upsample_block(
                    block_type=cfg.upsample_block_type,
                    in_channels=cfg.width_list[stage_id + 1],
                    out_channels=width if cfg.upsample_match_channel else cfg.width_list[stage_id + 1],
                    shortcut=cfg.upsample_shortcut,
                )
                stage.append(upsample_block)

            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            act = cfg.act[stage_id] if isinstance(cfg.act, list) else cfg.act
            stage.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=block_type,
                    norm=norm,
                    act=act,
                    input_width=(
                        width if cfg.upsample_match_channel else cfg.width_list[min(stage_id + 1, num_stages - 1)]
                    ),
                )
            )
            self.stages.insert(0, OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_decoder_project_out_block(
            in_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            out_channels=cfg.in_channels,
            factor=1 if cfg.depth_list[0] > 0 else 2,
            upsample_block_type=cfg.upsample_block_type,
            norm=cfg.out_norm,
            act=cfg.out_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in reversed(self.stages):
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x

class MGVQ(nn.Module):
    def __init__(self, cfg: MGVQConfig):
        super().__init__()
        self.cfg = cfg
        self.groups = cfg.codebook_groups
        self.codebook_size = cfg.codebook_size
        self.group_dim = cfg.latent_channels // self.groups
        self.encoder = Encoder(cfg.encoder)
        self.decoder = Decoder(cfg.decoder)
        self.quantize = VectorQuantizer(self.codebook_size, self.group_dim, 0.25, self.groups, True, True)
        self.quant_conv = nn.Conv2d(cfg.latent_channels, cfg.latent_channels, 1, groups=self.groups)
        self.post_quant_conv = nn.Conv2d(cfg.latent_channels, cfg.latent_channels, 1, groups=self.groups)

        if self.cfg.pretrained_path is not None:
            self.load_model()

    def load_model(self):
        if self.cfg.pretrained_source == "mgvq":
            state_dict = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)["state_dict"]
            self.load_state_dict(state_dict)
        else:
            raise NotImplementedError

    @property
    def spatial_compression_ratio(self) -> int:
        return 2 ** (self.decoder.num_stages - 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        quant = self.post_quant_conv(quant)
        x = self.decoder(quant)
        return x
    

    def decode_code(self, code_b, shape=None, channel_first=True, groups_to_use=0):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first, groups_to_use)
        x = self.decode(quant_b)
        return x
    

    def forward(self, input: torch.Tensor, global_step: int):
        quant, diff, _ = self.encode(input)
        x = self.decode(quant)
        return x, diff, quant


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, groups, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.groups = groups
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.n_e_group = self.n_e//self.groups

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer(f"codebook_used", nn.Parameter(torch.zeros(groups, 65536)))
        self.prob_alpha = 0.01
        self.register_buffer("embed_prob", torch.zeros(groups, self.n_e_group))

    
    def forward(self, z_groups):
        codebook_usage = torch.zeros(self.groups)
        vq_loss = torch.zeros(self.groups)
        commit_loss = torch.zeros(self.groups)
        z_q_list = []
        group_encoding_indices_ls = []
        # group_mask_prob_ls = [0.0, 0.2, 0.3, 0.4]
        group_mask_prob_ls = [0.0, 0.0, 0.0, 0.0] # second-stage finetuning
        group_mask_prob = torch.rand(1)
        for i in range(self.groups):

            z = z_groups[:, i*self.e_dim:i*self.e_dim+self.e_dim, ...]
            ''' for codebook usage initializing'''
            if self.show_usage:
                codebook_usage[i] = len(torch.unique(self.codebook_used[i])) / self.n_e_group
            ''' for coarse to fine masking '''
            if i > 0 and i < len(group_mask_prob_ls) and self.training:
                if group_mask_prob < group_mask_prob_ls[i]:
                    z_q_list.append(torch.zeros_like(z))
                    continue

            z = torch.einsum('b c h w -> b h w c', z).contiguous()
            z_flattened = z.view(-1, self.e_dim)

            if self.l2_norm:
                z = F.normalize(z, p=2, dim=-1)
                z_flattened = F.normalize(z_flattened, p=2, dim=-1)
                embedding = F.normalize(self.embedding.weight[i*self.n_e_group:(i+1)*self.n_e_group, :], p=2, dim=-1)
            else:
                embedding = self.embedding.weight[i*self.n_e_group:(i+1)*self.n_e_group, :]

            d = -torch.sum(z_flattened ** 2, dim=1, keepdim=True) - \
                torch.sum(embedding ** 2, dim=1) + 2 * \
                torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

            # encoding
            _, indices = d.sort(dim=1)
            # look up the closest point for the indices
            encoding_indices = indices[:,-1]
            z_q = embedding[encoding_indices].view(z.shape)
            group_encoding_indices_ls.append(encoding_indices)

            perplexity = None
            min_encodings = None
            vq_loss[i] = 0
            commit_loss[i] = 0
            entropy_loss = None
            codebook_usage[i] = 0

            if self.show_usage and self.training:
                cur_len = encoding_indices.shape[0]
                self.codebook_used[i][:-cur_len] = self.codebook_used[i][cur_len:].clone()
                self.codebook_used[i][-cur_len:] = encoding_indices
                codebook_usage[i] = len(torch.unique(self.codebook_used[i])) / self.n_e_group 
           # compute loss for embedding
            if self.training:
                vq_loss[i] = torch.mean((z_q - z.detach()) ** 2) 
                commit_loss[i] = self.beta * torch.mean((z_q.detach() - z) ** 2) 
                entropy_loss = 0

            # preserve gradients
            z_q = z + (z_q - z).detach()

            # reshape back to match original input shape
            z_q = torch.einsum('b h w c -> b c h w', z_q)
            z_q_list.append(z_q)

            # resample by used frequency EMA
            min_encodings = torch.nn.functional.one_hot(
                encoding_indices, num_classes=self.n_e_group
            ).to(dtype=z.dtype, device=z.device)
            avg_probs = torch.mean(min_encodings, dim=0) # prob of used codes
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) # e^-H(p) entropy of distribution p
            if self.training:
                embed_prob = self.embed_prob[i]
                updated_prob = torch.lerp(embed_prob, avg_probs, self.prob_alpha)
                self.embed_prob[i].copy_(updated_prob)
                norm_distance = F.softmax((d-d.max()).t(), dim=1)
                # check NaN / Inf
                if not torch.isfinite(norm_distance).all():
                    print("Warning: probs contain NaN or Inf, replacing with uniform distribution")
                    norm_distance = torch.ones_like(norm_distance) / norm_distance.numel()
                # avoid zero probability
                norm_distance = torch.clamp(norm_distance, min=1e-10)
                norm_distance /= torch.sum(norm_distance)
                # resample by distance
                prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                random_feat = z_flattened.detach()[prob]
                freq_prob = 1 - torch.exp((-(updated_prob*self.n_e_group*10)/self.prob_alpha)-0.001).unsqueeze(1).repeat(1, self.e_dim)
                self.embedding.weight.data[i*self.n_e_group:(i+1)*self.n_e_group, :] = self.embedding.weight.data[i*self.n_e_group:(i+1)*self.n_e_group, :] * freq_prob + random_feat * (1 - freq_prob)

        quant = torch.cat(z_q_list, dim=1)
        return quant, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, torch.stack(group_encoding_indices_ls, dim=-1))


    def get_codebook_entry(self, indices, shape=None, channel_first=True, groups_to_use=0):
        z_q_list = []
        for i in range(self.groups):
            if self.l2_norm:
                embedding = F.normalize(self.embedding.weight[i*self.n_e_group:(i+1)*self.n_e_group, :], p=2, dim=-1)
            else:
                embedding = self.embedding.weight[i*self.n_e_group:(i+1)*self.n_e_group, :]
            z_q = embedding[indices[:,:,i]]

            if i < groups_to_use:
                z_q_list.append(z_q)
            else:
                z_q_list.append(torch.zeros_like(z_q))
        quant = torch.cat(z_q_list, dim=-1)
        if shape is not None:
            if channel_first:
                quant = quant.reshape(shape[0], shape[2], shape[3], shape[1])
                quant = quant.permute(0, 3, 1, 2).contiguous()
            else:
                quant = quant.view(shape)
        return quant

def mgvq_f8c32(name: str, pretrained_path: str, codebook_size: int, codebook_groups: int) -> MGVQConfig:
    if name in ["mgvq-f8c32"]:
        cfg_str = (
            "latent_channels=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU] "
            "encoder.width_list=[128,256,512,512] encoder.depth_list=[0,4,8,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU] "
            "decoder.width_list=[128,256,512,512] decoder.depth_list=[0,5,10,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d] decoder.act=[relu,relu,relu,silu] "
            f"codebook_size={codebook_size} "
            f"codebook_groups={codebook_groups}"
        )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: MGVQConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(MGVQConfig), cfg))
    cfg.pretrained_path = pretrained_path
    return cfg

def mgvq_f16c32(name: str, pretrained_path: str, codebook_size: int, codebook_groups: int) -> MGVQConfig:
    if name in ["mgvq-f16c32"]:
        cfg_str = (
            "latent_channels=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU] "
            "encoder.width_list=[128,256,512,512,1024] encoder.depth_list=[0,4,8,2,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU] "
            "decoder.width_list=[128,256,512,512,1024] decoder.depth_list=[0,5,10,2,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu] "
            f"codebook_size={codebook_size} "
            f"codebook_groups={codebook_groups}"
        )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: MGVQConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(MGVQConfig), cfg))
    cfg.pretrained_path = pretrained_path
    return cfg

def mgvq_f32c32(name: str, pretrained_path: str, codebook_size: int, codebook_groups: int) -> MGVQConfig:
    if name in ["mgvq-f32c32"]:
        cfg_str = (
            "latent_channels=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024] encoder.depth_list=[0,4,8,2,2,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[0,5,10,2,2,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu] "
            f"codebook_size={codebook_size} "
            f"codebook_groups={codebook_groups}"
        )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: MGVQConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(MGVQConfig), cfg))
    cfg.pretrained_path = pretrained_path
    return cfg
