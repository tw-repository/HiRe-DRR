"""Train and evaluate MOI-Former on dyadic relationship datasets."""

import argparse
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, f1_score, recall_score
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from dataset import DEFAULT_ID_PROJECTION_PATH, Dataset


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.multiprocessing.set_sharing_strategy("file_system")

DATASET_CONFIGS = {
    "NoXi": {
        "data_dir": "xxx/NoXi_video_downSample/",
        "facial_feature_dir": (
            "NoXi_face_low_high_features.pkl"
        ),
        "body_feature_dir": (
            "NoXi_body_low_high_features.pkl"
        ),
        "audio_feature_dir": (
            "NoXi_audio_low_high_features.pkl"
        ),
        "text_dir": "xxx/NoXi_text.pkl",
        "train_list": "xxx/NoXi_train_bbox{}.pkl",
        "test_list": "xxx/NoXi_test_bbox{}.pkl",
        "save_dir": (
            "xxx/experimental_results/trained_models/NoXi"
        ),
        "labels": ["Stranger", "Acquaintance", "Friend", "Very good friend"],
    },
    "UDIVA": {
        "data_dir": "xxx/UDIVA_video_downSample",
        "facial_feature_dir": (
            "UDIVA_face_low_high_features.pkl"
        ),
        "body_feature_dir": (
            "UDIVA_body_low_high_features.pkl"
        ),
        "audio_feature_dir": (
            "UDIVA_audio_low_high_features.pkl"
        ),
        "text_dir": "xxx/UDIVA_text.pkl",
        "train_list": "xxx/UDIVA_train_bbox{}.pkl",
        "test_list": "xxx/UDIVA_test_bbox{}.pkl",
        "save_dir": (
            "xxx/experimental_results/trained_models/UDIVA"
        ),
        "labels": ["Known", "Unknown"],
    },
    "SeamInt": {
        "data_dir": "xxx/SeamInt_video_downSample/",
        "facial_feature_dir": (
            "SeamInt_face_low_high_features.pkl"
        ),
        "body_feature_dir": (
            "SeamInt_body_low_high_features.pkl"
        ),
        "audio_feature_dir": (
            "SeamInt_audio_low_high_features.pkl"
        ),
        "text_dir": (
            "SeamInt_text.pkl"
        ),
        "train_list": (
            "xxx/SeamInt_train_bbox{}.pkl"
        ),
        "test_list": (
            "xxx/SeamInt_test_bbox{}.pkl"
        ),
        "save_dir": (
             "xxx/experimental_results/trained_models/SeamInt"
        ),
        "labels": ["Stranger", "Acquaintance", "Friend", "Family", "Couple"],
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="Train MOI-Former")
    parser.add_argument("--mode", choices=DATASET_CONFIGS, default="NoXi")
    parser.add_argument("--person", choices=["A", "B"], default="B")
    parser.add_argument("--k-fold", dest="k_fold", type=int, default=3)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total-frames", dest="total_frames", type=int, default=10)
    parser.add_argument("--start-epoch", dest="start_epoch", type=int, default=1)
    parser.add_argument("--lr-step", dest="lr_step", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device, for example cuda, cuda:1 or cpu.",
    )
    parser.add_argument("--analyse-attention", action="store_true")

    for name in (
        "data_dir",
        "facial_feature_dir",
        "body_feature_dir",
        "audio_feature_dir",
        "text_dir",
        "train_list",
        "test_list",
        "save_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name)

    parser.add_argument("--id-projection-path", default=DEFAULT_ID_PROJECTION_PATH)
    parser.add_argument("--timesformer-path")
    parser.add_argument("--bert-path")
    parser.add_argument("--sentiment-model-path")
    return parser


def resolve_paths(args):
    config = DATASET_CONFIGS[args.mode].copy()
    for key in (
        "data_dir",
        "facial_feature_dir",
        "body_feature_dir",
        "audio_feature_dir",
        "text_dir",
        "train_list",
        "test_list",
        "save_dir",
    ):
        override = getattr(args, key)
        if override:
            config[key] = override

    config["train_list"] = config["train_list"].format(args.k_fold)
    config["test_list"] = config["test_list"].format(args.k_fold)
    return config


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pad_audio_sequence(sequence, max_length=1002):
    sequence = sequence[:max_length]
    return F.pad(sequence, (0, 0, 0, max_length - sequence.size(0)))


def _collate_participant(samples, offset):
    face_features = [sample[offset] for sample in samples]
    face_rois = [sample[offset + 1] for sample in samples]
    body_features = [sample[offset + 2] for sample in samples]
    body_rois = [sample[offset + 3] for sample in samples]
    audio_features = [sample[offset + 4] for sample in samples]
    texts = [sample[offset + 5] for sample in samples]

    def stack(records, key):
        return torch.stack([record[key] for record in records])

    def stack_audio(key):
        return torch.stack(
            [_pad_audio_sequence(record[key]) for record in audio_features]
        )

    return (
        stack(face_features, "low_facial_landmark"),
        stack(face_features, "low_head_pose"),
        stack(face_features, "high_gaze"),
        stack(face_features, "high_action_units"),
        torch.stack(face_rois).transpose(1, 2),
        stack(face_features, "high_valence_arousal"),
        stack(body_features, "low_body_pose"),
        stack(body_features, "high_body_openness"),
        stack(body_features, "high_orientation"),
        torch.stack(body_rois).transpose(1, 2),
        stack_audio("low_mfcc"),
        stack_audio("low_f0"),
        stack_audio("low_energy"),
        stack(audio_features, "high_emotion_embedding"),
        stack(audio_features, "high_speech_temporal"),
        stack(audio_features, "high_spectral_voice"),
        texts,
    )


def vg_collate(samples):
    """Collate the two participants while preserving the model input order."""
    participant_a = _collate_participant(samples, offset=0)
    participant_b = _collate_participant(samples, offset=7)
    relation_a = torch.as_tensor([sample[6] for sample in samples], dtype=torch.long)
    relation_b = torch.as_tensor([sample[13] for sample in samples], dtype=torch.long)
    position = torch.as_tensor([sample[14] for sample in samples], dtype=torch.long)
    video_length = torch.as_tensor([sample[15] for sample in samples], dtype=torch.long)
    return (
        *participant_a,
        *participant_b,
        relation_a,
        relation_b,
        position,
        video_length,
    )


class LabelVideoBalancedBatchSampler(Sampler):
    """Balance labels while avoiding repeated source videos within a batch."""

    def __init__(self, dataset, batch_size, drop_last=True, seed=42):
        if not hasattr(dataset, "sample_labels") or not hasattr(
            dataset, "sample_video_ids"
        ):
            raise ValueError("Dataset must expose sample_labels and sample_video_ids.")

        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.label_video_to_indices = defaultdict(lambda: defaultdict(list))

        for index, (label, video_id) in enumerate(
            zip(dataset.sample_labels, dataset.sample_video_ids)
        ):
            self.label_video_to_indices[label][video_id].append(index)

        self.labels = sorted(self.label_video_to_indices)
        if len(self.labels) < 2:
            raise ValueError("At least two classes are required for balanced sampling.")

        if drop_last:
            self.num_batches = len(dataset) // batch_size
        else:
            self.num_batches = math.ceil(len(dataset) / batch_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def _choose_labels_for_batch(self, rng):
        labels = self.labels.copy()
        rng.shuffle(labels)
        if self.batch_size <= len(labels):
            return labels[: self.batch_size]

        chosen = []
        while len(chosen) < self.batch_size:
            labels = self.labels.copy()
            rng.shuffle(labels)
            chosen.extend(labels)
        return chosen[: self.batch_size]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = {
            label: {
                video_id: rng.sample(indices, len(indices))
                for video_id, indices in videos.items()
            }
            for label, videos in self.label_video_to_indices.items()
        }

        for _ in range(self.num_batches):
            batch = []
            used_videos = set()

            for label in self._choose_labels_for_batch(rng):
                video_pool = pools[label]
                available = [
                    video_id for video_id, indices in video_pool.items() if indices
                ]
                if not available:
                    for video_id, indices in self.label_video_to_indices[label].items():
                        video_pool[video_id] = rng.sample(indices, len(indices))
                    available = list(video_pool)

                candidates = [
                    video_id for video_id in available if video_id not in used_videos
                ]
                chosen_video = rng.choice(candidates or available)
                batch.append(video_pool[chosen_video].pop())
                used_videos.add(chosen_video)

            yield batch


def _build_dataset(paths, args, transform, label_path):
    return Dataset(
        paths["data_dir"],
        label_path,
        paths["facial_feature_dir"],
        paths["body_feature_dir"],
        paths["audio_feature_dir"],
        paths["text_dir"],
        transform,
        total_frames=args.total_frames,
        id_projection_path=args.id_projection_path,
    )


def get_train_set(paths, args):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.7),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = _build_dataset(paths, args, transform, paths["train_list"])
    batch_sampler = LabelVideoBalancedBatchSampler(
        dataset,
        args.batch_size,
        drop_last=True,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        num_workers=args.workers,
        collate_fn=vg_collate,
        batch_sampler=batch_sampler,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    return loader


def get_test_set(paths, args):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = _build_dataset(paths, args, transform, paths["test_list"])
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=vg_collate,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )


def _prepare_batch(batch_data, device):
    model_inputs = [
        value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for value in batch_data[:34]
    ]
    relation_a, relation_b, position, video_length = (
        value.to(device, non_blocking=True) for value in batch_data[34:]
    )
    return model_inputs, relation_a, relation_b, position, video_length


def _target_for_mode(relation_a, relation_b, mode, person):
    return relation_b if mode == "NoXi" and person == "B" else relation_a


def softmax_np(values, axis=1):
    values = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def calculate_single_batch_metrics(
    true_labels,
    pred_scores,
    num_classes,
    from_logits=True,
):
    """Compute class recall, UAR, macro-F1 and mAP over one prediction set."""
    true_labels = np.asarray(true_labels, dtype=int).reshape(-1)
    pred_scores = np.asarray(pred_scores)
    pred_probs = softmax_np(pred_scores) if from_logits else pred_scores
    pred_labels = np.argmax(pred_probs, axis=1)
    labels = np.arange(num_classes)

    recall_per_class = recall_score(
        true_labels,
        pred_labels,
        labels=labels,
        average=None,
        zero_division=0,
    )
    recall_dict = dict(enumerate(recall_per_class))
    uar = float(np.mean(recall_per_class))
    f1 = f1_score(
        true_labels,
        pred_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    one_hot = np.zeros((len(true_labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(true_labels)), true_labels] = 1
    average_precisions = [
        average_precision_score(one_hot[:, class_id], pred_probs[:, class_id])
        if one_hot[:, class_id].any()
        else np.nan
        for class_id in range(num_classes)
    ]
    mean_average_precision = (
        float(np.nanmean(average_precisions))
        if not np.all(np.isnan(average_precisions))
        else float("nan")
    )
    return recall_dict, uar, f1, mean_average_precision


def calculate_metrics(true_labels, pred_scores, num_classes, from_logits=True):
    return calculate_single_batch_metrics(
        true_labels,
        pred_scores,
        num_classes,
        from_logits,
    )


@torch.no_grad()
def validate(loader, model, device, args, num_classes):
    model.eval()
    true_labels, prediction_scores = [], []

    for batch_data in tqdm(loader, desc="Validate", leave=False):
        model_inputs, relation_a, relation_b, position, video_length = _prepare_batch(
            batch_data,
            device,
        )
        target = _target_for_mode(
            relation_a,
            relation_b,
            args.mode,
            args.person,
        )
        logits, _ = model(*model_inputs, position, video_length)
        true_labels.append(target.cpu().numpy())
        prediction_scores.append(logits.cpu().numpy())

    return calculate_metrics(
        np.concatenate(true_labels),
        np.concatenate(prediction_scores),
        num_classes,
        from_logits=True,
    )


def get_relation_loss_config(mode):
    if mode == "NoXi":
        return {
            "binary_splits": [
                ([0], [1, 2, 3], 1.00),
                ([1], [2, 3], 0.75),
                ([2], [3], 0.50),
            ],
            "focal_gamma": 0.0,
            "hier_weight": 0.15,
            "rank_weight": 0.0,
            "macro_average": False,
        }
    if mode == "UDIVA":
        return {
            "binary_splits": [],
            "focal_gamma": 1.0,
            "hier_weight": 0.0,
            "rank_weight": 0.15,
            "macro_average": True,
        }
    if mode in {"SeamInt", "Seamless Interaction"}:
        return {
            "binary_splits": [
                ([0], [1, 2, 3, 4], 1.00),
                ([1], [2, 3, 4], 0.40),
                ([2], [3, 4], 0.20),
            ],
            "focal_gamma": 0.0,
            "hier_weight": 0.05,
            "rank_weight": 0.0,
            "macro_average": True,
        }
    raise ValueError(f"Unknown mode: {mode}")


def classification_loss(logits, target, focal_gamma, macro_average=False):
    log_prob = F.log_softmax(logits, dim=-1)
    target_log_prob = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
    loss_per_sample = -target_log_prob

    if focal_gamma > 0:
        loss_per_sample = (1.0 - target_log_prob.exp()).pow(
            focal_gamma
        ) * loss_per_sample
    if not macro_average:
        return loss_per_sample.mean()

    class_losses = [
        loss_per_sample[target == class_id].mean()
        for class_id in torch.unique(target)
    ]
    return torch.stack(class_losses).mean()


def conditional_hierarchy_loss(logits, target, binary_splits):
    if not binary_splits:
        return logits.new_tensor(0.0)

    weighted_losses = []
    total_weight = 0.0
    for left_classes, right_classes, node_weight in binary_splits:
        left_classes = torch.as_tensor(left_classes, device=logits.device)
        right_classes = torch.as_tensor(right_classes, device=logits.device)
        left_mask = (target[:, None] == left_classes[None, :]).any(dim=1)
        right_mask = (target[:, None] == right_classes[None, :]).any(dim=1)
        valid_mask = left_mask | right_mask
        if not valid_mask.any():
            continue

        left_score = torch.logsumexp(logits.index_select(1, left_classes), dim=1)
        right_score = torch.logsumexp(logits.index_select(1, right_classes), dim=1)
        branch_loss = F.binary_cross_entropy_with_logits(
            (right_score - left_score)[valid_mask],
            right_mask[valid_mask].float(),
            reduction="none",
        )
        branch_target = right_mask[valid_mask]
        balanced_losses = [
            branch_loss[mask].mean()
            for mask in (~branch_target, branch_target)
            if mask.any()
        ]
        weighted_losses.append(node_weight * torch.stack(balanced_losses).mean())
        total_weight += node_weight

    if not weighted_losses:
        return logits.new_tensor(0.0)
    return torch.stack(weighted_losses).sum() / total_weight


def binary_ranking_loss(logits, target, positive_class=1):
    negative_class = 1 - positive_class
    relation_score = logits[:, positive_class] - logits[:, negative_class]
    positive_score = relation_score[target == positive_class]
    negative_score = relation_score[target == negative_class]
    if positive_score.numel() == 0 or negative_score.numel() == 0:
        return logits.new_tensor(0.0)

    differences = positive_score[:, None] - negative_score[None, :]
    return F.softplus(-differences).mean()


def hierarchical_relation_loss(logits, target, mode):
    target = target.flatten().long()
    config = get_relation_loss_config(mode)
    loss_cls = classification_loss(
        logits,
        target,
        config["focal_gamma"],
        config["macro_average"],
    )
    loss_hier = conditional_hierarchy_loss(
        logits,
        target,
        config["binary_splits"],
    )
    loss_rank = (
        binary_ranking_loss(logits, target, positive_class=1)
        if mode == "UDIVA"
        else logits.new_tensor(0.0)
    )
    total_loss = (
        loss_cls
        + config["hier_weight"] * loss_hier
        + config["rank_weight"] * loss_rank
    )
    return total_loss, {
        "loss_cls": loss_cls.detach(),
        "loss_hier": loss_hier.detach(),
        "loss_rank": loss_rank.detach(),
        "loss_total": total_loss.detach(),
    }


def _normalize01(values, eps=1e-8):
    value_range = values.max() - values.min()
    if value_range < eps:
        return torch.full_like(values, 0.5)
    return (values - values.min()) / (value_range + eps)


def analyze_top5_attention(
    attention_dicts,
    query_name="mixture_interaction_readout",
    topk=5,
):
    """Summarise low/high and homogeneous/heterogeneous routing preferences."""
    if not attention_dicts:
        raise ValueError("attention_dicts cannot be empty.")

    query_index = attention_dicts[0]["query_names"].index(query_name)
    token_sum = None
    total_count = 0
    for attention in attention_dicts:
        cross_attention = attention["decoder_cross_attn"][-1]
        batch_size, num_heads = cross_attention.shape[:2]
        batch_sum = cross_attention[:, :, query_index, :].sum(dim=(0, 1))
        token_sum = batch_sum if token_sum is None else token_sum + batch_sum
        total_count += batch_size * num_heads

    token_attention = token_sum / total_count
    low_attention = token_attention[:16]
    high_attention = token_attention[16:]
    low_score = low_attention.sum().item()
    high_score = high_attention.sum().item()
    dominant_level = "low" if low_score >= high_score else "high"
    dominant_attention = low_attention if dominant_level == "low" else high_attention
    token_offset = 0 if dominant_level == "low" else 16

    normalised = _normalize01(dominant_attention)
    top_values, top_indices = torch.topk(normalised, k=min(topk, 16))
    modalities = ["face", "body", "audio", "text"]
    top_interactions = []
    for rank, (pair_index, score) in enumerate(
        zip(top_indices.tolist(), top_values.tolist()),
        start=1,
    ):
        a_index, b_index = divmod(pair_index, 4)
        top_interactions.append(
            {
                "rank": rank,
                "level": dominant_level,
                "pair_index": pair_index,
                "token_index": token_offset + pair_index,
                "a_modal": modalities[a_index],
                "b_modal": modalities[b_index],
                "interaction_type": (
                    "homogeneous" if a_index == b_index else "heterogeneous"
                ),
                "raw_attention": dominant_attention[pair_index].item(),
                "score": score,
            }
        )

    return {
        "query_name": query_name,
        "level_attention_raw": {"low": low_score, "high": high_score},
        "dominant_level": dominant_level,
        "top_interactions": top_interactions,
        "low_matrix_norm": _normalize01(low_attention).reshape(4, 4),
        "high_matrix_norm": _normalize01(high_attention).reshape(4, 4),
        "modalities": modalities,
        "topk": topk,
    }


def _print_interaction_matrix(matrix, title):
    print(f"\n{title}")
    print(f"{'':12s}{'B-Face':>10s}{'B-Body':>10s}{'B-Audio':>10s}{'B-Text':>10s}")
    for row_name, row in zip(["A-Face", "A-Body", "A-Audio", "A-Text"], matrix):
        values = "".join(f"{value.item():10.4f}" for value in row)
        print(f"{row_name:12s}{values}")


def print_top5_attention(analysis):
    level = analysis["dominant_level"]
    print(f"\nDominant feature level: {level}")
    print(f"Top-{analysis['topk']} {level}-level interactions")
    for item in analysis["top_interactions"]:
        print(
            f"{item['rank']:02d} | A_{item['a_modal']} - B_{item['b_modal']} "
            f"| {item['interaction_type']} | score={item['score']:.6f}"
        )
    _print_interaction_matrix(analysis["low_matrix_norm"], "Low-level interactions")
    _print_interaction_matrix(analysis["high_matrix_norm"], "High-level interactions")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.val = value
        self.sum += value * count
        self.count += count
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, initial_lr, step):
    learning_rate = initial_lr * (0.1 ** (epoch // step))
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def _compact_attention(diagnostics):
    return {
        "query_names": diagnostics["query_names"],
        "decoder_cross_attn": [
            diagnostics["decoder_cross_attn"][-1].detach().cpu()
        ],
    }


def _checkpoint_name(args, epoch, uar, mean_average_precision):
    person = f"_{args.person}" if args.mode == "NoXi" else ""
    return (
        f"{args.mode}_K{args.k_fold}_{epoch}{person}_"
        f"{uar:.2%}_{mean_average_precision:.2%}.pth.tar"
    )


def train(train_loader, test_loader, model, optimizer, device, args, config):
    best_uar = float("-inf")
    best_map = float("-inf")
    os.makedirs(config["save_dir"], exist_ok=True)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        model.train()
        model.visual_features.eval()
        model.low_text_feature.eval()
        model.high_text_feature.eval()
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args.lr, args.lr_step)
        losses = AverageMeter()
        epoch_attention = []

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            model_inputs, relation_a, relation_b, position, video_length = (
                _prepare_batch(batch_data, device)
            )
            target = _target_for_mode(
                relation_a,
                relation_b,
                args.mode,
                args.person,
            )

            optimizer.zero_grad(set_to_none=True)
            logits, diagnostics = model(*model_inputs, position, video_length)
            loss, _ = hierarchical_relation_loss(logits, target, args.mode)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), target.size(0))

            if args.analyse_attention:
                epoch_attention.append(_compact_attention(diagnostics))

        print(
            f"Train | epoch {epoch}/{args.start_epoch + args.epochs - 1} "
            f"| loss {losses.avg:.4f}"
        )
        if epoch_attention:
            print_top5_attention(analyze_top5_attention(epoch_attention))

        recall, uar, f1, mean_average_precision = validate(
            test_loader,
            model,
            device,
            args,
            len(config["labels"]),
        )
        print(f"Recall: {recall}")
        print(
            f"Validate | UAR {uar:.2%} | F1 {f1:.2%} "
            f"| mAP {mean_average_precision:.2%}"
        )

        if uar > best_uar or mean_average_precision > best_map:
            best_uar = max(best_uar, uar)
            best_map = max(best_map, mean_average_precision)
            checkpoint_path = os.path.join(
                config["save_dir"],
                _checkpoint_name(args, epoch, uar, mean_average_precision),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": {
                        "uar": uar,
                        "f1": f1,
                        "mAP": mean_average_precision,
                    },
                },
                checkpoint_path,
            )


def init_network(num_classes, args, device):
    from network_MOI import (
        DEFAULT_BERT_PATH,
        DEFAULT_SENTIMENT_MODEL_PATH,
        DEFAULT_TIMESFORMER_PATH,
        Network,
    )

    model = Network(
        num_classes,
        data_name=args.mode,
        timesformer_path=args.timesformer_path or DEFAULT_TIMESFORMER_PATH,
        bert_path=args.bert_path or DEFAULT_BERT_PATH,
        sentiment_model_path=(
            args.sentiment_model_path or DEFAULT_SENTIMENT_MODEL_PATH
        ),
        num_frames=args.total_frames,
    ).to(device)

    for module in (
        model.visual_features,
        model.low_text_feature,
        model.high_text_feature,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = False

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    return model, trainable_parameters


def main():
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    set_random_seed(args.seed)
    config = resolve_paths(args)
    device = torch.device(args.device)
    for name, value in sorted(vars(args).items()):
        print(f"{name}: {value}")

    print("Creating data loaders...")
    train_loader = get_train_set(config, args)
    test_loader = get_test_set(config, args)

    print("Loading network...")
    model, parameters = init_network(len(config["labels"]), args, device)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=5e-4,
    )
    train(train_loader, test_loader, model, optimizer, device, args, config)


if __name__ == "__main__":
    main()
