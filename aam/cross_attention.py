from .pipeline import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import torch
import itertools

from .hook import ObjectHooker, AggregateHooker


class DiffusionAttentionHooker(AggregateHooker[Attention]):
    def __init__(
            self,
            pipeline: StableDiffusionPipeline,
    ):
        model = pipeline.unet
        blocks_list = []
        up_names = ['up'] * len(model.up_blocks)
        down_names = ['down'] * len(model.up_blocks)

        for unet_block, name in itertools.chain(
            zip(model.up_blocks, up_names),
            zip(model.down_blocks, down_names),
            zip([model.mid_block], 'mid')
        ):
            if 'CrossAttn' in unet_block.__class__.__name__:
                blocks = []

                for spatial_transformer in unet_block.attentions:
                    for transformer_block in spatial_transformer.transformer_blocks:
                        blocks.append(transformer_block.attn2)
                        blocks.append(transformer_block.attn1)

                    blocks = [b for idx, b in enumerate(blocks)]
                    blocks_list.extend(blocks)

        self.modules = [
            UNetCrossAttentionHooker(x) for idx, x in enumerate(blocks_list)
        ]

        super().__init__(self.modules)


class UNetCrossAttentionHooker(ObjectHooker[Attention]):
    def __init__(self, module: Attention):
        super().__init__(module)

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            modifier_idx=None,
            phrase_idx=None,
            encoder_hidden_states=None,
            attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_self_attn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            is_self_attn = True

        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        if is_self_attn:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)

        else:
            key_0 = attn.to_k(encoder_hidden_states[0])
            key_1 = attn.to_k(encoder_hidden_states[1])
            value = attn.to_v(encoder_hidden_states[0])

            query = attn.head_to_batch_dim(query)
            key_0 = attn.head_to_batch_dim(key_0)
            key_1 = attn.head_to_batch_dim(key_1)
            value = attn.head_to_batch_dim(value)

            attention_probs_0 = attn.get_attention_scores(query, key_0, attention_mask)
            attention_probs_1 = attn.get_attention_scores(query, key_1, attention_mask)

            # init mapper & alpha maps
            mapper = torch.arange(0, 77, 1).to('cuda')
            alpha = torch.ones((1, 1, 77)).to('cuda')
            alpha_2 = torch.ones((1, 1, 77)).to('cuda')

            mapper[modifier_idx] = -1
            alpha[0, 0, modifier_idx] = 0
            phrase_weight = len(phrase_idx)
            for i in phrase_idx: alpha_2[0, 0, i] = 1 - 1 / phrase_weight

            attention_probs_1 = attention_probs_1 * (1 - alpha_2)
            for i in range(2, len(phrase_idx) + 1):
                attention_probs_1[:, :, 1] += attention_probs_1[:, :, i]
            attention_probs = attention_probs_0[:, :, mapper] * alpha
            attention_probs[:, :, modifier_idx] = attention_probs_1[:, :, 1]

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def _hook_impl(self):
        self.module.set_processor(self)


hook = DiffusionAttentionHooker
