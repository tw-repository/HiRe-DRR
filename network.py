"""Mixture-of-Interaction Transformer for dyadic relationship recognition."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timesformer.models.vit import TimeSformer
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_TIMESFORMER_PATH = (
    "xxx/preTrainedModels/"
    "TimeSformer_divST_96x4_224_K400.pyth"
)
DEFAULT_BERT_PATH = (
    "xxx/preTrainedModels/"
    "bert-base-uncased"
)
DEFAULT_SENTIMENT_MODEL_PATH = (
    "xxx/preTrainedModels/"
    "twitter-roberta-base-sentiment-latest"
)
DEFAULT_TOP_K = 5
VISUAL_FEATURE_DIM = 512


def _module_device(module):
    return next(module.parameters()).device


class PersonPairVisualEncoder(nn.Module):
    def __init__(self, pretrained_model=DEFAULT_TIMESFORMER_PATH, num_frames=10):
        super().__init__()
        self.feature_extractor = TimeSformer(
            img_size=224,
            num_classes=VISUAL_FEATURE_DIM,
            num_frames=num_frames,
            attention_type="divided_space_time",
            pretrained_model=pretrained_model,
        )

    def forward(self, face_A, person_A, face_B, person_B):
        face_a = self.feature_extractor(face_A)
        face_b = self.feature_extractor(face_B)
        person_a = self.feature_extractor(person_A)
        person_b = self.feature_extractor(person_B)

        return face_a, face_b, person_a, person_b


class LowLevelTextFeature(nn.Module):
    def __init__(self, tokenizer, language_model):
        super().__init__()
        self.tokenizer = tokenizer
        self.language_model = language_model

    @torch.no_grad()
    def forward(self, text_list):
        self.language_model.eval()
        tokens = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        device = _module_device(self.language_model)
        tokens = {name: value.to(device) for name, value in tokens.items()}
        return self.language_model(**tokens).last_hidden_state[:, 0, :]


class HighLevelTextFeature(nn.Module):
    def __init__(self, model_path=DEFAULT_SENTIMENT_MODEL_PATH):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_hidden_states=True,
        )

    @torch.no_grad()
    def forward(self, text_list):
        self.model.eval()
        inputs = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        device = _module_device(self.model)
        inputs = {name: value.to(device) for name, value in inputs.items()}
        hidden_states = self.model(**inputs).hidden_states[-1]
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        return (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(
            dim=1
        ).clamp_min(1)


def _temporal_stats(features):
    """Concatenate temporal mean, deviation and first-order motion."""
    batch_size, steps = features.shape[:2]
    features = features.reshape(batch_size, steps, -1)
    mean = features.mean(dim=1)
    std = features.std(dim=1, unbiased=False)
    motion = (
        (features[:, 1:] - features[:, :-1]).abs().mean(dim=1)
        if steps > 1
        else torch.zeros_like(mean)
    )
    return torch.cat([mean, std, motion], dim=-1)


class FaceFeatureFusion(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()

        self.low_proj = nn.Sequential(
            nn.Linear(478 * 3 * 3 + 6 * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        self.high_proj = nn.Sequential(
            nn.Linear(52 * 3 + 8 * 3 + 2 * 3 + 512, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(
        self,
        low_facial_landmark,
        low_head_pose,
        high_action_units,
        high_gaze,
        high_valence_arousal,
        face_visual
    ):
        low_landmark = _temporal_stats(low_facial_landmark)
        low_pose = _temporal_stats(low_head_pose)

        high_au = _temporal_stats(high_action_units)
        high_gaze = _temporal_stats(high_gaze)
        high_va = _temporal_stats(high_valence_arousal)

        low_face = self.low_proj(
            torch.cat([low_landmark, low_pose], dim=-1)
        )

        high_face = self.high_proj(
            torch.cat([high_au, high_gaze, high_va, face_visual], dim=-1)
        )
        return low_face, high_face


class BodyFeatureFusion(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()

        self.low_proj = nn.Sequential(
            nn.Linear(17 * 3 * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        self.high_proj = nn.Sequential(
            nn.Linear(8 * 3 + 6 * 3 + 512, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(
        self,
        low_body_pose,
        high_body_openness,
        high_orientation,
        body_visual
    ):
        low_pose = _temporal_stats(low_body_pose)
        high_open = _temporal_stats(high_body_openness)
        high_orient = _temporal_stats(high_orientation)
        low_body = self.low_proj(low_pose)

        high_body = self.high_proj(
            torch.cat([high_open, high_orient, body_visual], dim=-1)
        )

        return low_body, high_body


class AudioFeatureFusion(nn.Module):
    def __init__(
        self,
        output_dim=512,
        emotion_dim=1024,
        speech_temporal_dim=4,
        spectral_voice_dim=3
    ):
        super().__init__()

        self.low_proj = nn.Sequential(
            nn.Linear((20 + 1 + 1) * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        self.high_proj = nn.Sequential(
            nn.Linear(
                emotion_dim + speech_temporal_dim + spectral_voice_dim,
                output_dim
            ),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def _add_batch_dim_low(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    def _add_batch_dim_high(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def forward(
        self,
        low_mfcc,
        low_f0,
        low_energy,
        high_emotion_embedding,
        high_speech_temporal,
        high_spectral_voice,
    ):
        low_mfcc = self._add_batch_dim_low(low_mfcc)
        low_f0 = self._add_batch_dim_low(low_f0)
        low_energy = self._add_batch_dim_low(low_energy)
        high_emotion_embedding = self._add_batch_dim_high(high_emotion_embedding)
        high_speech_temporal = self._add_batch_dim_high(high_speech_temporal)
        high_spectral_voice = self._add_batch_dim_high(high_spectral_voice)

        low_mfcc_feat = _temporal_stats(low_mfcc)
        low_f0_feat = _temporal_stats(low_f0)
        low_energy_feat = _temporal_stats(low_energy)

        low_audio = self.low_proj(
            torch.cat(
                [low_mfcc_feat, low_f0_feat, low_energy_feat],
                dim=-1
            )
        )

        high_audio = self.high_proj(
            torch.cat(
                [
                    high_emotion_embedding,
                    high_speech_temporal,
                    high_spectral_voice
                ],
                dim=-1
            )
        )

        return low_audio, high_audio


class RelativeTemporalEncoding(nn.Module):
    def __init__(
        self,
        model_dim=2048,
        max_relative_positions=512,
        position_offset=1.0
    ):
        super().__init__()

        if model_dim % 2 != 0:
            raise ValueError("model_dim must be even for sinusoidal encoding.")

        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / model_dim)
        )

        self.model_dim = model_dim
        self.max_relative_positions = float(max_relative_positions)
        self.position_offset = float(position_offset)
        self.register_buffer("div_term", div_term)

    def forward(self, position, video_length):
        position = torch.as_tensor(
            position,
            dtype=self.div_term.dtype,
            device=self.div_term.device
        ).reshape(-1, 1)

        video_length = torch.as_tensor(
            video_length,
            dtype=self.div_term.dtype,
            device=self.div_term.device
        ).reshape(-1, 1)

        relative_position = (
            (position - self.position_offset)
            / (video_length - 1.0).clamp_min(1.0)
        ).clamp(0.0, 1.0)

        relative_index = relative_position * (
            self.max_relative_positions - 1.0
        )
        angles = relative_index * self.div_term.unsqueeze(0)

        temporal_embedding = torch.zeros(
            position.size(0),
            self.model_dim,
            dtype=angles.dtype,
            device=angles.device
        )
        temporal_embedding[:, 0::2] = torch.sin(angles)
        temporal_embedding[:, 1::2] = torch.cos(angles)

        return temporal_embedding


class RelationAwareInteractionRouter(nn.Module):
    def __init__(
        self,
        d_model=2048,
        num_tokens=32,
        top_k=6,
        dropout=0.3,
        temperature=1.0,
        noise_std=0.1
    ):
        super().__init__()

        if not 1 <= top_k <= num_tokens:
            raise ValueError(
                f"top_k must be in [1, {num_tokens}], but got {top_k}."
            )
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

        hidden_dim = max(d_model // 4, 128)

        self.num_tokens = num_tokens
        self.top_k = top_k
        self.temperature = float(temperature)
        self.noise_std = float(noise_std)

        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        if num_tokens != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} interaction tokens, "
                f"but received {num_tokens}."
            )

        global_context = x.mean(
            dim=1,
            keepdim=True
        ).expand(-1, num_tokens, -1)

        router_input = torch.cat(
            [x, global_context],
            dim=-1
        )

        router_logits = self.scorer(router_input).squeeze(-1)

        routing_logits = router_logits
        if self.training and self.noise_std > 0.0:
            routing_logits = routing_logits + torch.randn_like(
                routing_logits
            ) * self.noise_std

        dense_weights = F.softmax(
            routing_logits / self.temperature,
            dim=-1
        )

        topk_weights, topk_indices = torch.topk(
            dense_weights,
            k=self.top_k,
            dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1,
            keepdim=True
        ).clamp_min(1e-8)

        sparse_weights = torch.zeros_like(dense_weights)
        sparse_weights.scatter_(
            dim=-1,
            index=topk_indices,
            src=topk_weights
        )

        return {
            "router_logits": router_logits,
            "dense_weights": dense_weights,
            "sparse_weights": sparse_weights,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            "token_mask": sparse_weights > 0
        }


class SparseInteractionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=2048,
        num_heads=8,
        num_tokens=32,
        top_k=6,
        dropout=0.3,
        ffn_ratio=4,
        router_temperature=1.0,
        router_noise_std=0.1
    ):
        super().__init__()

        self.top_k = top_k

        self.router = RelationAwareInteractionRouter(
            d_model=d_model,
            num_tokens=num_tokens,
            top_k=top_k,
            dropout=dropout,
            temperature=router_temperature,
            noise_std=router_noise_std
        )

        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

        self.interaction_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout)
        )

    @staticmethod
    def _gather_tokens(x, indices):
        """
        x:       [B, N, D]
        indices: [B, K]
        return:  [B, K, D]
        """
        gather_index = indices.unsqueeze(-1).expand(
            -1, -1, x.size(-1)
        )
        return torch.gather(x, dim=1, index=gather_index)

    @staticmethod
    def _scatter_tokens(token_bank, indices, selected_tokens):
        scatter_index = indices.unsqueeze(-1).expand(
            -1, -1, token_bank.size(-1)
        )
        return token_bank.scatter(
            dim=1,
            index=scatter_index,
            src=selected_tokens
        )

    def forward(self, x):
        """
        x: [B, 32, D]

        return:
            x:            [B, 32, D]
            attn:         [B, heads, K, 32]
        """
        routing_info = self.router(x)
        topk_indices = routing_info["topk_indices"]
        topk_weights = routing_info["topk_weights"]

        selected_tokens = self._gather_tokens(
            x,
            topk_indices
        )

        attn_out, attn = self.interaction_attn(
            self.query_norm(selected_tokens),
            self.memory_norm(x),
            self.memory_norm(x),
            need_weights=True,
            average_attn_weights=False
        )

        selected_context = selected_tokens + self.attn_dropout(attn_out)
        selected_candidate = selected_context + self.ffn(
            self.ffn_norm(selected_context)
        )

        # The continuous gate lets routing weights influence the token update.
        update_gate = 1.0 + topk_weights.unsqueeze(-1)
        selected_updated = selected_tokens + update_gate * (
            selected_candidate - selected_tokens
        )

        x = self._scatter_tokens(
            x,
            topk_indices,
            selected_updated
        )

        routing_info["selected_tokens_before"] = selected_tokens
        routing_info["selected_tokens_after"] = selected_updated

        return x, attn, routing_info


class MixtureInteractionDecoder(nn.Module):
    def __init__(
        self,
        d_model=2048,
        num_tokens=32,
        top_k=6,
        dropout=0.3,
        temperature=1.0,
        noise_std=0.1
    ):
        super().__init__()

        self.router = RelationAwareInteractionRouter(
            d_model=d_model,
            num_tokens=num_tokens,
            top_k=top_k,
            dropout=dropout,
            temperature=temperature,
            noise_std=noise_std
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, memory):
        """
        memory: [B, 32, D]

        return:
            relation_feat: [B, D]
        """
        routing_info = self.router(memory)
        sparse_weights = routing_info["sparse_weights"]

        relation_feat = torch.sum(
            memory * sparse_weights.unsqueeze(-1),
            dim=1
        )
        relation_feat = self.output_norm(relation_feat)

        return relation_feat, routing_info


class MOIFormer(nn.Module):
    def __init__(
        self,
        num_classes,
        input_dim=768,
        proj_dim=512,
        model_dim=2048,
        num_heads=8,
        encoder_layers=2,
        dropout=0.3,
        top_k=6,
        router_temperature=1.0,
        router_noise_std=0.1
    ):
        super().__init__()

        self.model_name = "MOI-Former"
        self.full_model_name = "Mixture-of-Interaction Transformer"
        self.modalities = ["face", "body", "audio", "text"]
        self.level_names = ["low", "high"]
        self.num_interaction_tokens = 32
        self.top_k = top_k

        self.proj_dim = proj_dim
        self.model_dim = model_dim

        self.low_text_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        self.high_text_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        self.temporal_encoding = RelativeTemporalEncoding(
            model_dim=model_dim
        )
        self.input_fusion_norm = nn.LayerNorm(model_dim)

        # These indices are metadata for routing visualisation only.
        token_indices = torch.arange(self.num_interaction_tokens)
        level_ids = token_indices // 16
        pair_ids = token_indices % 16
        row_ids = pair_ids // 4
        col_ids = pair_ids % 4
        type_ids = (row_ids != col_ids).long()

        self.register_buffer("level_ids", level_ids.long())
        self.register_buffer("pair_ids", pair_ids.long())
        self.register_buffer("type_ids", type_ids.long())

        self.encoder_layers = nn.ModuleList([
            SparseInteractionEncoderLayer(
                d_model=model_dim,
                num_heads=num_heads,
                num_tokens=self.num_interaction_tokens,
                top_k=top_k,
                dropout=dropout,
                router_temperature=router_temperature,
                router_noise_std=router_noise_std
            )
            for _ in range(encoder_layers)
        ])

        self.mixture_decoder = MixtureInteractionDecoder(
            d_model=model_dim,
            num_tokens=self.num_interaction_tokens,
            top_k=top_k,
            dropout=dropout,
            temperature=router_temperature,
            noise_std=router_noise_std
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )

    def _stack_modalities(self, face, body, audio, text):
        """
        face/body/audio/text: [B, 512]
        return: [B, 4, 512]
        """
        return torch.stack([face, body, audio, text], dim=1)

    def _build_16_interactions(self, A_feat, B_feat):
        """
        A_feat: [B, 4, 512]
        B_feat: [B, 4, 512]

        return:
            tokens: [B, 16, 2048]
        """
        batch_size, num_modalities, feat_dim = A_feat.shape

        A_expand = A_feat.unsqueeze(2).expand(
            batch_size,
            num_modalities,
            num_modalities,
            feat_dim
        )
        B_expand = B_feat.unsqueeze(1).expand(
            batch_size,
            num_modalities,
            num_modalities,
            feat_dim
        )

        consistency = A_expand * B_expand
        difference = torch.abs(A_expand - B_expand)

        tokens = torch.cat(
            [
                A_expand,
                B_expand,
                consistency,
                difference
            ],
            dim=-1
        )

        return tokens.reshape(
            batch_size,
            num_modalities * num_modalities,
            feat_dim * 4
        )

    def forward(
        self,
        low_face_A, high_face_A,
        low_face_B, high_face_B,
        low_body_A, high_body_A,
        low_body_B, high_body_B,
        low_audio_A, high_audio_A,
        low_audio_B, high_audio_B,
        low_text_A, low_text_B,
        high_text_A, high_text_B,
        position, video_length
    ):
        low_text_A = self.low_text_proj(low_text_A)
        low_text_B = self.low_text_proj(low_text_B)
        high_text_A = self.high_text_proj(high_text_A)
        high_text_B = self.high_text_proj(high_text_B)

        low_A = self._stack_modalities(
            low_face_A,
            low_body_A,
            low_audio_A,
            low_text_A
        )
        low_B = self._stack_modalities(
            low_face_B,
            low_body_B,
            low_audio_B,
            low_text_B
        )

        high_A = self._stack_modalities(
            high_face_A,
            high_body_A,
            high_audio_A,
            high_text_A
        )
        high_B = self._stack_modalities(
            high_face_B,
            high_body_B,
            high_audio_B,
            high_text_B
        )

        low_tokens_raw = self._build_16_interactions(low_A, low_B)
        high_tokens_raw = self._build_16_interactions(high_A, high_B)

        interaction_tokens_raw = torch.cat(
            [low_tokens_raw, high_tokens_raw],
            dim=1
        )

        temporal_embedding = self.temporal_encoding(
            position,
            video_length
        )

        x = self.input_fusion_norm(
            interaction_tokens_raw
            + temporal_embedding.unsqueeze(1)
        )
        encoder_input = x

        # Each encoder layer reroutes over the complete interaction-token bank.
        encoder_attn_list = []
        encoder_routing_list = []
        token_bank_list = [x]

        for layer in self.encoder_layers:
            x, enc_attn, layer_routing = layer(x)
            encoder_attn_list.append(enc_attn)
            encoder_routing_list.append(layer_routing)
            token_bank_list.append(x)

        memory = x

        relation_feat, final_routing = self.mixture_decoder(memory)
        logits = self.classifier(relation_feat)
        final_sparse_weights = final_routing["sparse_weights"]

        layer_sparse_weights = [
            item["sparse_weights"]
            for item in encoder_routing_list
        ]
        layer_topk_indices = [
            item["topk_indices"]
            for item in encoder_routing_list
        ]
        layer_topk_weights = [
            item["topk_weights"]
            for item in encoder_routing_list
        ]

        if layer_sparse_weights:
            routing_frequency = torch.stack(
                [weights > 0 for weights in layer_sparse_weights],
                dim=0
            ).float().mean(dim=(0, 1))
        else:
            routing_frequency = torch.zeros(
                self.num_interaction_tokens,
                device=memory.device,
                dtype=memory.dtype
            )

        attn_dict = {
            "model_name": self.model_name,
            "full_model_name": self.full_model_name,

            "low_tokens_raw": low_tokens_raw,
            "high_tokens_raw": high_tokens_raw,
            "interaction_tokens_raw": interaction_tokens_raw,
            "encoder_input": encoder_input,

            "interaction_tokens": memory,
            "memory": memory,
            "token_bank_per_layer": token_bank_list,
            "relation_feat": relation_feat,
            "temporal_embedding": temporal_embedding,

            "encoder_attn": encoder_attn_list,
            "encoder_routing": encoder_routing_list,
            "encoder_router_logits": [
                item["router_logits"]
                for item in encoder_routing_list
            ],
            "encoder_dense_token_weights": [
                item["dense_weights"]
                for item in encoder_routing_list
            ],
            "encoder_sparse_token_weights": layer_sparse_weights,
            "encoder_topk_token_indices": layer_topk_indices,
            "encoder_topk_token_weights": layer_topk_weights,
            "routing_frequency": routing_frequency,

            "router_logits": final_routing["router_logits"],
            "dense_token_weights": final_routing["dense_weights"],
            "token_weights": final_sparse_weights,
            "token_mask": final_routing["token_mask"],
            "topk_token_indices": final_routing["topk_indices"],
            "topk_token_weights": final_routing["topk_weights"],
            "token_usage": final_sparse_weights.mean(dim=0),
            "num_interaction_tokens": self.num_interaction_tokens,
            "top_k": self.top_k,

            # Compatibility aliases retained for existing analysis scripts.
            "dense_expert_weights": final_routing["dense_weights"],
            "expert_weights": final_sparse_weights,
            "expert_mask": final_routing["token_mask"],
            "topk_expert_indices": final_routing["topk_indices"],
            "topk_expert_weights": final_routing["topk_weights"],
            "expert_usage": final_sparse_weights.mean(dim=0),
            "num_experts": self.num_interaction_tokens,

            "query_output": relation_feat.unsqueeze(1),
            "query_weight_logits": final_routing["router_logits"],
            "query_weights": final_sparse_weights,
            "decoder_self_attn": [],
            "decoder_cross_attn": [
                final_sparse_weights.unsqueeze(1).unsqueeze(1)
            ],

            "level_ids": self.level_ids,
            "pair_ids": self.pair_ids,
            "type_ids": self.type_ids,

            "token_names": self.get_token_names(),
            "expert_names": self.get_token_names(),
            "pair_names": self.get_pair_names(),
            "level_names": self.level_names,
            "query_names": ["mixture_interaction_readout"],
            "modality_names": self.modalities,
            "type_names": ["homogeneous", "heterogeneous"]
        }

        return logits, attn_dict

    def get_pair_names(self):
        return [
            f"A_{a}_with_B_{b}"
            for a in self.modalities
            for b in self.modalities
        ]

    def get_token_names(self):
        pair_names = self.get_pair_names()

        return [
            f"{level}_{pair}"
            for level in self.level_names
            for pair in pair_names
        ]


class Network(nn.Module):
    def __init__(
        self,
        num_classes,
        data_name=None,
        timesformer_path=DEFAULT_TIMESFORMER_PATH,
        bert_path=DEFAULT_BERT_PATH,
        sentiment_model_path=DEFAULT_SENTIMENT_MODEL_PATH,
        num_frames=10,
        top_k=DEFAULT_TOP_K,
    ):
        super().__init__()
        self.data_name = data_name
        self.visual_features = PersonPairVisualEncoder(timesformer_path, num_frames)

        bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert_model = AutoModel.from_pretrained(bert_path)
        self.low_text_feature = LowLevelTextFeature(bert_tokenizer, bert_model)
        self.high_text_feature = HighLevelTextFeature(sentiment_model_path)

        self.face_fusion = FaceFeatureFusion(output_dim=512)
        self.body_fusion = BodyFeatureFusion(output_dim=512)
        self.audio_fusion = AudioFeatureFusion(output_dim=512, emotion_dim=1024)
        self.interaction_model = MOIFormer(
            num_classes=num_classes,
            input_dim=768,
            proj_dim=512,
            model_dim=2048,
            num_heads=8,
            top_k=top_k,
            encoder_layers=2,
            dropout=0.3,
        )

    def forward(
        self,
        low_face_A_landmark,
        low_face_A_headPose,
        high_face_A_gaze,
        high_face_A_actionUnits,
        face_A_ROI,
        high_face_A_VA,
        low_body_A_pose,
        high_body_A_openness,
        high_body_A_orientation,
        body_A_ROI,
        low_audio_A_mfcc,
        low_audio_A_f0,
        low_audio_A_energy,
        high_audio_A_emotion_embedding,
        high_audio_A_high_speech_temporal,
        high_audio_A_high_spectral_voice,
        text_A,
        low_face_B_landmark,
        low_face_B_headPose,
        high_face_B_gaze,
        high_face_B_actionUnits,
        face_B_ROI,
        high_face_B_VA,
        low_body_B_pose,
        high_body_B_openness,
        high_body_B_orientation,
        body_B_ROI,
        low_audio_B_mfcc,
        low_audio_B_f0,
        low_audio_B_energy,
        high_audio_B_emotion_embedding,
        high_audio_B_high_speech_temporal,
        high_audio_B_high_spectral_voice,
        text_B,
        position,
        video_length,
        data_name=None,
    ):
        face_visual_A, face_visual_B, body_visual_A, body_visual_B = (
            self.visual_features(face_A_ROI, body_A_ROI, face_B_ROI, body_B_ROI)
        )

        low_face_A, high_face_A = self.face_fusion(
            low_face_A_landmark,
            low_face_A_headPose,
            high_face_A_actionUnits,
            high_face_A_gaze,
            high_face_A_VA,
            face_visual_A,
        )
        low_face_B, high_face_B = self.face_fusion(
            low_face_B_landmark,
            low_face_B_headPose,
            high_face_B_actionUnits,
            high_face_B_gaze,
            high_face_B_VA,
            face_visual_B,
        )
        low_body_A, high_body_A = self.body_fusion(
            low_body_A_pose,
            high_body_A_openness,
            high_body_A_orientation,
            body_visual_A,
        )
        low_body_B, high_body_B = self.body_fusion(
            low_body_B_pose,
            high_body_B_openness,
            high_body_B_orientation,
            body_visual_B,
        )
        low_audio_A, high_audio_A = self.audio_fusion(
            low_audio_A_mfcc,
            low_audio_A_f0,
            low_audio_A_energy,
            high_audio_A_emotion_embedding,
            high_audio_A_high_speech_temporal,
            high_audio_A_high_spectral_voice,
        )
        low_audio_B, high_audio_B = self.audio_fusion(
            low_audio_B_mfcc,
            low_audio_B_f0,
            low_audio_B_energy,
            high_audio_B_emotion_embedding,
            high_audio_B_high_speech_temporal,
            high_audio_B_high_spectral_voice,
        )
        low_text_A = self.low_text_feature(text_A)
        low_text_B = self.low_text_feature(text_B)
        high_text_A = self.high_text_feature(text_A)
        high_text_B = self.high_text_feature(text_B)

        return self.interaction_model(
            low_face_A,
            high_face_A,
            low_face_B,
            high_face_B,
            low_body_A,
            high_body_A,
            low_body_B,
            high_body_B,
            low_audio_A,
            high_audio_A,
            low_audio_B,
            high_audio_B,
            low_text_A,
            low_text_B,
            high_text_A,
            high_text_B,
            position,
            video_length,
        )

