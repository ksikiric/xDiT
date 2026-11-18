import functools
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux3 import (
    Flux3Attention,
    Flux3Transformer2DModel,
    _get_qkv_projections,
)

from xfuser.core.distributed import (
    get_ring_parallel_world_size,
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from xfuser.core.distributed.attention_a2a import (
    fused_a2a_pad_multiple,
    get_fused_a2a_hadamard_placement,
    get_fused_a2a_mode,
    get_fused_a2a_profile,
)
from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.model_executor.layers import usp_sdma
from xfuser.model_executor.layers.aiter_int8_linear import (
    quantize_shared_input,
    rope_pairs,
    xFuserAiterInt8Linear,
)
from xfuser.model_executor.layers.fused_qk_norm_rope_wan_flydsl import wan_flydsl_hadamard
from xfuser.model_executor.layers.monokernel import (
    MONOKERNEL,
    MONOKERNEL_HADAMARD,
    MONOKERNEL_PROFILES,
    monokernel_rows_multiple,
)
from xfuser.model_executor.layers.usp import PROCESS_GROUP, USP, _ft_c_output_all_to_all, attention
from xfuser.model_executor.models.transformers.transformers_utils import (
    chunk_and_pad_sequence,
    gather_and_unpad,
)


def _attention_a2a_active() -> bool:
    """Whether sharded stages send Q/K/V through the Attention A2A input hop."""
    return (
        get_fused_a2a_mode() > 0
        and get_ulysses_parallel_world_size() > 1
        and get_ring_parallel_world_size() == 1
        and not get_runtime_state().runtime_config.use_sparseattn_head_balance
    )


def _sp_pad(seq_len: int, sp_world_size: int) -> int:
    """Rows that pad a sharded stage to equal chunks, which the Attention A2A layouts
    (and the monokernel's tiled K/V layouts) need in whole row tiles per rank."""
    multiple = sp_world_size
    if _attention_a2a_active():
        multiple = fused_a2a_pad_multiple(sp_world_size)
        if MONOKERNEL:
            multiple = math.lcm(multiple, sp_world_size * monokernel_rows_multiple(get_fused_a2a_profile()))
    return (multiple - seq_len % multiple) % multiple


def _project(linear, hidden_states, shared):
    return linear(hidden_states) if shared is None else linear(hidden_states, quantized=shared)


class xFuserFlux3AttnProcessor:
    """Flux3AttnProcessor with the attention call routed through USP.

    Each sequence stage (text, image and reference-image modes, then the joint
    blocks) shares one instance. Before a stage runs, the transformer sets
    ``valid_kv_len`` to the stage's real length when the SP pad made the sequence
    longer, so the pad keys are dropped after the all-to-all, and ``local`` when
    every rank holds the whole stream, so attention runs without collectives.

    With Attention A2A active, sharded stages send Q/K/V through its quantized input hop.
    With XFUSER_MONOKERNEL=1 and a fused AITER INT8 Q/K/V projection, AITER's pre-attention
    monokernel runs the projection, the Q/K prep and the hop in one launch; the transformer
    then also sets ``rope_pairs``, the stage's RoPE tables per rotation pair.
    """

    def __init__(self):
        self.valid_kv_len: Optional[int] = None
        self.local = False
        self.rope_pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def __call__(
        self,
        attn: Flux3Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError("FLUX.3 attention takes no mask.")

        activation_dtype = hidden_states.dtype
        attention_a2a = not self.local and _attention_a2a_active()
        if attention_a2a and image_rotary_emb is not None and self._monokernel_eligible(attn):
            hidden_states = self._monokernel(attn, hidden_states, image_rotary_emb)
        else:
            query, key, value = self._projections(attn, hidden_states)

            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))
            query, key, value = self._norm_rope(attn, query, key, value, image_rotary_emb)
            if attention_a2a:
                if get_fused_a2a_hadamard_placement() != "none":
                    query = wan_flydsl_hadamard(query)
                    key = wan_flydsl_hadamard(key)
                # A fused Q/K/V projection leaves V a strided slice; the input hop reads whole rows.
                value = value.contiguous()

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            if self.local:
                hidden_states = attention(query, key, value)
            else:
                attention_kwargs = None if self.valid_kv_len is None else {"valid_kv_len": self.valid_kv_len}
                hidden_states = USP(
                    query,
                    key,
                    value,
                    attn_layer=attn,
                    attention_kwargs=attention_kwargs,
                    combine_qkv_a2a=True,
                    attention_a2a_enabled=attention_a2a,
                )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(activation_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    @staticmethod
    def _projections(attn, hidden_states):
        if attn.fused_projections:
            return _get_qkv_projections(attn, hidden_states)
        linears = (attn.to_q, attn.to_k, attn.to_v)
        shared = quantize_shared_input(linears, hidden_states)
        return tuple(_project(linear, hidden_states, shared) for linear in linears)

    @staticmethod
    def _monokernel_eligible(attn) -> bool:
        return (
            MONOKERNEL
            and attn.fused_projections
            and isinstance(attn.to_qkv, xFuserAiterInt8Linear)
            and get_fused_a2a_hadamard_placement() in MONOKERNEL_HADAMARD
            and get_fused_a2a_profile() in MONOKERNEL_PROFILES
        )

    def _monokernel(self, attn, hidden_states, image_rotary_emb):
        """[b, heads, s, d] attention output; FLUX.3's prep is a per-head RMSNorm, interleaved
        RoPE, no bias and the profile's Q/K Hadamard."""
        qkv = attn.to_qkv
        x_quantized, x_scale = torch.ops.xfuser.aiter_int8_quantize_rows(
            hidden_states.reshape(-1, qkv.in_features).contiguous()
        )
        pairs = self.rope_pairs
        if pairs is None:
            pairs = rope_pairs(image_rotary_emb[0], image_rotary_emb[1], attn.head_dim)
        eps = attn.norm_q.eps if attn.norm_q.eps is not None else torch.finfo(torch.float32).eps
        output = torch.ops.xfuser.monokernel_attention(
            x_quantized,
            x_scale,
            qkv.weight_int8,
            qkv.weight_scale,
            None,
            attn.norm_q.weight,
            attn.norm_k.weight,
            pairs[0],
            pairs[1],
            attn.heads,
            "rms_per_head",
            "interleaved",
            get_fused_a2a_hadamard_placement() == "preprocess",
            eps,
            self.valid_kv_len,
            get_fused_a2a_profile(),
            x_quantized.shape[0],
            x_quantized.shape[1],
            "int8",
        )
        return _ft_c_output_all_to_all(output.transpose(1, 2))

    @staticmethod
    def _norm_rope(attn, query, key, value, image_rotary_emb):
        """QK norm and RoPE of [b, s, heads, d] projections, for any subset of heads."""
        # QK norm and RoPE in fp32 with one cast back, as upstream.
        query = attn.norm_q(query.float())
        key = attn.norm_k(key.float())

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        return query.to(value.dtype), key.to(value.dtype), value


@usp_sdma.sdma_processor(xFuserFlux3AttnProcessor)
class xFuserFlux3SdmaAttnProcessor(xFuserFlux3AttnProcessor):
    """xFuserFlux3AttnProcessor under --ulysses_a2a sdma --ulysses_sdma_overlap_qkv.

    The QK norm and RoPE act per head, so Q/K/V are projected one head group at a time and
    each group's exchange overlaps the next group's projection
    (usp_sdma.grouped_qkv_attention). Calls that path does not cover take the base path.
    """

    sdma_grouped_qkv = True

    def __call__(
        self,
        attn: Flux3Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not self.local and attention_mask is None:
            grouped = usp_sdma.grouped_qkv_attention(
                attn,
                hidden_states,
                lambda q, k, v: self._norm_rope(attn, q, k, v, image_rotary_emb),
                backend=None,
                attention_kwargs=None if self.valid_kv_len is None else {"valid_kv_len": self.valid_kv_len},
                group=PROCESS_GROUP.ULYSSES_PG,
            )
            if grouped is not None:
                hidden_states = grouped.transpose(1, 2).flatten(2, 3).to(hidden_states.dtype)
                hidden_states = attn.to_out[0](hidden_states)
                return attn.to_out[1](hidden_states)
        return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)


def flux3_attn_modules(transformer) -> list[torch.nn.Module]:
    """Return every FLUX.3 attention module that executes USP, in execution order."""
    blocks = (
        *transformer.txt_mode_blocks,
        *transformer.mode_blocks["image"],
        *transformer.mode_blocks["image_cond"],
        *transformer.single_transformer_blocks,
    )
    return [block.attn for block in blocks]


class xFuserFlux3Transformer2DWrapper(Flux3Transformer2DModel):
    """Sequence-parallel FLUX.3 transformer.

    Every sequence is padded at its end to a multiple of the SP degree (of whole row
    tiles per rank under Attention A2A) and split evenly across ranks, so after the
    Ulysses all-to-all the pad is a uniform suffix that ``valid_kv_len`` removes:

    - Per-mode blocks: the text, image and reference-image streams are attended
      separately. The image and reference streams are sharded, then gathered. A
      text stream of at most ``replicate_text_max_tokens`` runs whole on every
      rank instead, which skips its collectives for a little repeated compute.
    - Joint blocks: ``[text, image, reference]`` is sharded as one sequence. A
      rank's chunk can straddle the text/content boundary, so each block gets
      the chunk's text part (possibly empty) as ``encoder_hidden_states`` and the
      rest as ``hidden_states``. The blocks run unmodified; with one rank this is
      exactly the upstream call.
    """

    # Measured on 8x MI350X at 1024² (u8): replicating 80 or 400 text tokens saves about
    # 1.5 ms per eager forward of 38 ms; under HIP graphs the difference vanishes.
    replicate_text_max_tokens = 512

    # ConfigMixin picks the config keys to pass from this signature. Wrapping upstream's
    # __init__ shows it upstream's parameters; behind a bare **kwargs every key, flow_map
    # included, would fall back to its default.
    @functools.wraps(Flux3Transformer2DModel.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._txt_processor = xFuserFlux3AttnProcessor()
        self._image_processor = xFuserFlux3AttnProcessor()
        self._image_cond_processor = xFuserFlux3AttnProcessor()
        self._joint_processor = xFuserFlux3AttnProcessor()
        stages = (
            (self.txt_mode_blocks, self._txt_processor),
            (self.mode_blocks["image"], self._image_processor),
            (self.mode_blocks["image_cond"], self._image_cond_processor),
            (self.single_transformer_blocks, self._joint_processor),
        )
        for blocks, processor in stages:
            for block in blocks:
                block.attn.processor = processor
        register_fp8_comms_eligible_modules(self, flux3_attn_modules(self))

    @staticmethod
    def _shard(
        sp_rank: int,
        sp_world_size: int,
        pad: int,
        hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ):
        hidden_states = chunk_and_pad_sequence(hidden_states, sp_rank, sp_world_size, pad, dim=1)
        temb_mod = chunk_and_pad_sequence(temb_mod, sp_rank, sp_world_size, pad, dim=1)
        rotary_emb = tuple(chunk_and_pad_sequence(r, sp_rank, sp_world_size, pad, dim=0) for r in rotary_emb)
        return hidden_states, temb_mod, rotary_emb

    def _monokernel_rope_pairs(self, rotary_emb: Tuple[torch.Tensor, torch.Tensor]):
        """A stage's RoPE tables for the monokernel, built once per forward instead of per block."""
        if not (MONOKERNEL and _attention_a2a_active()):
            return None
        return rope_pairs(rotary_emb[0], rotary_emb[1], self.config.attention_head_dim)

    def _run_mode_blocks(
        self,
        blocks: torch.nn.ModuleList,
        processor: xFuserFlux3AttnProcessor,
        hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        local: bool = False,
    ) -> torch.Tensor:
        processor.local = local
        if local:
            processor.valid_kv_len = None
            for block in blocks:
                hidden_states = block(hidden_states, temb_mod, rotary_emb)
            return hidden_states

        sp_world_size = get_sequence_parallel_world_size()
        seq_len = hidden_states.shape[1]
        pad = _sp_pad(seq_len, sp_world_size)
        processor.valid_kv_len = seq_len if pad else None

        hidden_states, temb_mod, rotary_emb = self._shard(
            get_sequence_parallel_rank(), sp_world_size, pad, hidden_states, temb_mod, rotary_emb
        )
        processor.rope_pairs = self._monokernel_rope_pairs(rotary_emb)
        for block in blocks:
            hidden_states = block(hidden_states, temb_mod, rotary_emb)
        return gather_and_unpad(hidden_states, pad, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        image_cond_hidden_states: Optional[torch.Tensor] = None,
        image_cond_ids: Optional[torch.Tensor] = None,
        target_timestep: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        if image_cond_hidden_states is not None and image_cond_ids is None:
            raise ValueError("`image_cond_hidden_states` requires `image_cond_ids`.")
        if image_cond_hidden_states is None and image_cond_ids is not None:
            raise ValueError("`image_cond_ids` was passed without `image_cond_hidden_states`.")
        if (target_timestep is not None) != bool(self.config.flow_map):
            raise ValueError("`target_timestep` is required by, and only by, a flow-map transformer (`flow_map=True`).")

        batch_size, image_seq_len = hidden_states.shape[:2]
        dtype = hidden_states.dtype

        # 1-4. Timestep embeddings, modulations, input projections and RoPE, as upstream.
        image_temb = self.time_proj(timestep.to(dtype) * 1000)
        image_temb = self.timestep_embedder(image_temb.to(dtype))[:, None, :]
        zero_temb = self.time_proj(torch.zeros(batch_size, device=hidden_states.device, dtype=dtype))
        zero_temb = self.timestep_embedder(zero_temb.to(dtype))[:, None, :]
        if target_timestep is not None:
            target_temb = self.time_proj(target_timestep.to(dtype) * 1000)
            target_temb = self.target_timestep_embedder(target_temb.to(dtype))[:, None, :]
            image_temb = image_temb + target_temb
            zero_temb = zero_temb + target_temb
        image_temb = image_temb.expand(batch_size, image_seq_len, -1)
        zero_temb_act = F.silu(zero_temb)

        txt_temb_act = zero_temb_act.expand(batch_size, encoder_hidden_states.shape[1], -1)
        txt_early_mod = self.early_stream_modulations["txt"](txt_temb_act)
        txt_single_mod = self.single_stream_modulations["txt"](txt_temb_act)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        txt_rotary_emb = self.pos_embed(txt_ids)

        image_temb_act = F.silu(image_temb)
        image_early_mod = self.early_stream_modulations["image"](image_temb_act)
        image_single_mod = self.single_stream_modulations["image"](image_temb_act)
        hidden_states = self.x_embedders["image"](hidden_states)
        image_rotary_emb = self.pos_embed(img_ids)

        if image_cond_hidden_states is not None:
            cond_temb_act = zero_temb_act.expand(batch_size, image_cond_hidden_states.shape[1], -1)
            cond_early_mod = self.early_stream_modulations["image_cond"](cond_temb_act)
            cond_single_mod = self.single_stream_modulations["image_cond"](cond_temb_act)
            image_cond_hidden_states = self.x_embedders["image_cond"](image_cond_hidden_states)
            cond_rotary_emb = self.pos_embed(image_cond_ids)

        # 5. Per-mode blocks, each mode attended on its own; short text runs whole on every rank.
        replicate_text = (
            get_sequence_parallel_world_size() > 1 and encoder_hidden_states.shape[1] <= self.replicate_text_max_tokens
        )
        encoder_hidden_states = self._run_mode_blocks(
            self.txt_mode_blocks,
            self._txt_processor,
            encoder_hidden_states,
            txt_early_mod,
            txt_rotary_emb,
            local=replicate_text,
        )
        hidden_states = self._run_mode_blocks(
            self.mode_blocks["image"], self._image_processor, hidden_states, image_early_mod, image_rotary_emb
        )
        streams = [
            (encoder_hidden_states, txt_single_mod, txt_rotary_emb),
            (hidden_states, image_single_mod, image_rotary_emb),
        ]
        if image_cond_hidden_states is not None:
            image_cond_hidden_states = self._run_mode_blocks(
                self.mode_blocks["image_cond"],
                self._image_cond_processor,
                image_cond_hidden_states,
                cond_early_mod,
                cond_rotary_emb,
            )
            streams.append((image_cond_hidden_states, cond_single_mod, cond_rotary_emb))

        # 6. Joint blocks over [text, image, reference], sharded as one sequence.
        txt_seq_len = encoder_hidden_states.shape[1]
        joint_states = torch.cat([s[0] for s in streams], dim=1)
        joint_mod = torch.cat([s[1] for s in streams], dim=1)
        joint_rotary_emb = (
            torch.cat([s[2][0] for s in streams], dim=0),
            torch.cat([s[2][1] for s in streams], dim=0),
        )
        joint_seq_len = joint_states.shape[1]

        sp_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()
        pad = _sp_pad(joint_seq_len, sp_world_size)
        self._joint_processor.valid_kv_len = joint_seq_len if pad else None
        joint_states, joint_mod, joint_rotary_emb = self._shard(
            sp_rank, sp_world_size, pad, joint_states, joint_mod, joint_rotary_emb
        )
        self._joint_processor.rope_pairs = self._monokernel_rope_pairs(joint_rotary_emb)

        # Text tokens of this rank's chunk: the leading part, possibly none.
        chunk_len = joint_states.shape[1]
        local_txt_len = min(max(txt_seq_len - sp_rank * chunk_len, 0), chunk_len)
        context_states = joint_states[:, :local_txt_len]
        content_states = joint_states[:, local_txt_len:]
        context_mod = joint_mod[:, :local_txt_len]
        content_mod = joint_mod[:, local_txt_len:]

        for block in self.single_transformer_blocks:
            context_states, content_states = block(
                hidden_states=content_states,
                encoder_hidden_states=context_states,
                temb_mod=content_mod,
                temb_mod_context=context_mod,
                image_rotary_emb=joint_rotary_emb,
            )

        # 7. Output layers on this rank's image tokens, which follow its text tokens. They
        #    are per token, so only the output channels need gathering. Every image token
        #    has the same modulation, so any rows of image_temb serve.
        local_image_len = min(max(txt_seq_len + image_seq_len - sp_rank * chunk_len, 0), chunk_len) - local_txt_len
        hidden_states = content_states[:, :local_image_len]
        hidden_states = self.norm_out(hidden_states, image_temb[:, :local_image_len])
        output = self.proj_out(hidden_states)
        output = F.pad(output, (0, 0, local_txt_len, chunk_len - local_txt_len - local_image_len))
        output = gather_and_unpad(output, pad, dim=1)[:, txt_seq_len : txt_seq_len + image_seq_len]

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
