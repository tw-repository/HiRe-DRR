"""Dataset utilities for dyadic social-relationship recognition."""

import math
import os
import pickle as pkl
from functools import lru_cache

import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


DEFAULT_ID_PROJECTION_PATH = (
    "xxx/seamless_interaction/id_proj.pkl"
)
VIDEO_NAME_PARTS = {"NoXi": 2, "UDIVA": 3, "SeamInt": 4}


def _video_part_count(data_name):
    try:
        return VIDEO_NAME_PARTS[data_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {data_name}") from exc


def _paired_clip_key(key):
    parts = key.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid clip key: {key}")
    return "_".join(parts[:-2] + [parts[-1]])


def _normalise_label(label):
    if isinstance(label, (list, tuple)):
        label = label[0]
    elif hasattr(label, "item"):
        label = label.item()
    return int(label)


@lru_cache(maxsize=None)
def _load_id_projection(path):
    with open(path, "rb") as file:
        return pkl.load(file, encoding="latin1")


@lru_cache(maxsize=None)
def _load_role_to_person(path):
    reverse_mapping = {}
    for person_key, role_id in _load_id_projection(path).items():
        parts = person_key.split("_")
        reverse_mapping[("_".join(parts[:3]), str(role_id))] = parts[3]
    return reverse_mapping


def group_boxes_by_category(
    image_names,
    face_boxes,
    person_boxes,
    relation_classes,
    data_name,
    total_frame=50,
):
    """Group frame annotations into fixed-length, non-overlapping clips."""
    num_name_parts = _video_part_count(data_name)
    videos = {}

    for filename in image_names:
        video_id = "_".join(filename.split("_")[:num_name_parts])
        videos.setdefault(video_id, []).append(
            {
                "image_name": filename,
                "face_box": face_boxes[filename],
                "person_box": person_boxes[filename],
                "relation_class": relation_classes[filename],
            }
        )

    grouped_data = {}
    clip_counts = {}
    for video_id, items in videos.items():
        complete_clip_count = len(items) // total_frame
        clip_counts[video_id] = complete_clip_count

        for clip_index in range(complete_clip_count):
            start = clip_index * total_frame
            clip = items[start : start + total_frame]
            grouped_data[f"{video_id}_{clip_index + 1}"] = {
                "image_name": [item["image_name"] for item in clip],
                "face_boxes": [item["face_box"] for item in clip],
                "person_boxes": [item["person_box"] for item in clip],
                "relation_classes": [item["relation_class"] for item in clip],
            }

    return grouped_data, clip_counts


def sync_dictionaries(dict_a, dict_b):
    """Keep only clips shared by both participants and align their order."""
    keyed_a = {_paired_clip_key(key): key for key in dict_a}
    keyed_b = {_paired_clip_key(key): key for key in dict_b}
    common_keys = sorted(keyed_a.keys() & keyed_b.keys())

    synced_a = {keyed_a[key]: dict_a[keyed_a[key]] for key in common_keys}
    synced_b = {keyed_b[key]: dict_b[keyed_b[key]] for key in common_keys}
    return synced_a, synced_b


def align_info(input_dict, data_name):
    """Remove the final clip from every video, preserving the original mapping."""
    _video_part_count(data_name)
    final_clip_keys = {}

    for key in input_dict:
        parts = key.split("_")
        video_id = "_".join(parts[:-2])
        clip_index = int(parts[-1])
        current = final_clip_keys.get(video_id)
        if current is None or clip_index > current[0]:
            final_clip_keys[video_id] = (clip_index, key)

    excluded = {value[1] for value in final_clip_keys.values()}
    return {key: value for key, value in input_dict.items() if key not in excluded}


def select_frame_indices(sequence_length, num_samples):
    if sequence_length <= 0:
        raise ValueError("Cannot sample frames from an empty sequence.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    interval = math.ceil(sequence_length / num_samples)
    return [min(index * interval, sequence_length - 1) for index in range(num_samples)]


def select_frames(input_list, num_samples):
    indices = select_frame_indices(len(input_list), num_samples)
    return [input_list[index] for index in indices]


def clip_level_data(input_dict, transform, image_dir, total_frames):
    """Load synchronised face and body crops for one clip."""
    indices = select_frame_indices(len(input_dict["image_name"]), total_frames)
    image_names = [input_dict["image_name"][index] for index in indices]
    face_boxes = [input_dict["face_boxes"][index] for index in indices]
    person_boxes = [input_dict["person_boxes"][index] for index in indices]

    face_clip, body_clip = image_crop(
        image_names,
        image_dir,
        face_boxes,
        person_boxes,
        transform,
    )
    label = _normalise_label(input_dict["relation_classes"][0])
    return face_clip, body_clip, label


def restore_original_frame_name(
    current_frame_name,
    id_projection_path=DEFAULT_ID_PROJECTION_PATH,
):
    """Restore a SeamInt role identifier (0/1) to its original person ID."""
    directory = os.path.dirname(current_frame_name)
    parts = os.path.basename(current_frame_name).split("_")
    if len(parts) < 5:
        return current_frame_name

    video_key = "_".join(parts[:3])
    role_id = parts[3]
    original_person_id = _load_role_to_person(id_projection_path).get(
        (video_key, role_id)
    )
    if original_person_id is None:
        return current_frame_name

    parts[3] = original_person_id
    restored_name = "_".join(parts)
    return os.path.join(directory, restored_name) if directory else restored_name


def image_crop(
    image_name_list,
    image_dir,
    face_box_list,
    person_box_list,
    transform,
):
    cropped_faces, cropped_bodies = [], []
    dataset_prefix = os.path.basename(os.path.normpath(image_dir)).split("_")[0]

    for image_name, face_box, person_box in zip(
        image_name_list,
        face_box_list,
        person_box_list,
    ):
        if dataset_prefix not in {"NoXi", "UDIVA"}:
            image_name = restore_original_frame_name(image_name)

        with Image.open(os.path.join(image_dir, image_name)) as image:
            image = image.convert("RGB")
            cropped_faces.append(transform(image.crop(tuple(face_box))))
            cropped_bodies.append(transform(image.crop(tuple(person_box))))

    return torch.stack(cropped_faces), torch.stack(cropped_bodies)


def rename_keys(aligned_a, aligned_b, data_audio_text):
    """Replace original video identifiers with deterministic video_XXX IDs."""
    standard_keys = set(aligned_a) | set(aligned_b)
    video_names = sorted({"_".join(key.split("_")[:-2]) for key in standard_keys})
    video_map = {
        video_name: f"video_{index:03d}"
        for index, video_name in enumerate(video_names, start=1)
    }

    def rename_aligned(source):
        renamed = {}
        for old_key, value in source.items():
            parts = old_key.split("_")
            video_name = "_".join(parts[:-2])
            renamed[f"{video_map[video_name]}_{'_'.join(parts[-2:])}"] = value
        return renamed

    renamed_audio_text = {}
    for old_key, value in data_audio_text.items():
        if old_key not in standard_keys:
            continue
        parts = old_key.split("_")
        video_name = "_".join(parts[:-2])
        if video_name in video_map:
            new_key = f"{video_map[video_name]}_{'_'.join(parts[-2:])}"
            renamed_audio_text[new_key] = value

    return rename_aligned(aligned_a), rename_aligned(aligned_b), renamed_audio_text


def restore_audio_feature_key_to_role_id(
    audio_feature,
    id_proj_path=DEFAULT_ID_PROJECTION_PATH,
):
    """Replace SeamInt person IDs in audio keys with their paired role IDs."""
    id_projection = _load_id_projection(id_proj_path)
    restored = {}

    for key, value in audio_feature.items():
        parts = key.split("_")
        if len(parts) >= 4:
            person_key = "_".join(parts[:4])
            if person_key in id_projection:
                parts[3] = str(id_projection[person_key])
                key = "_".join(parts)
        restored[key] = value

    return restored


def filter_aligned_info_by_audio_feature(aligned_info, audio_feature):
    return {key: value for key, value in aligned_info.items() if key in audio_feature}


def _infer_dataset_name(image_dir):
    path_parts = os.path.normpath(image_dir).split(os.sep)
    for part in reversed(path_parts):
        prefix = part.split("_")[0]
        if prefix in {"NoXi", "UDIVA"}:
            return prefix
        if part.lower() == "seamless_interaction":
            return "SeamInt"
    raise ValueError(f"Cannot infer dataset name from image directory: {image_dir}")


class Dataset(TorchDataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        facial_feature_dir,
        body_feature_dir,
        audio_feature_dir,
        text_dir,
        transform,
        total_frames,
        id_projection_path=DEFAULT_ID_PROJECTION_PATH,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.total_frames = total_frames
        self.data_name = _infer_dataset_name(image_dir)

        self.data = self._load_pickle(label_dir)
        self.facial_feature = self._load_pickle(facial_feature_dir)
        self.body_feature = self._load_pickle(body_feature_dir)
        self.audio_feature = self._load_pickle(audio_feature_dir)
        self.text = self._load_pickle(text_dir)

        if self.data_name == "SeamInt":
            self.audio_feature = restore_audio_feature_key_to_role_id(
                self.audio_feature,
                id_projection_path,
            )

        participant_data = {
            "A": {"names": [], "face": {}, "body": {}, "relation": {}},
            "B": {"names": [], "face": {}, "body": {}, "relation": {}},
        }
        for item in self.data:
            participant, face_box, body_box, relation = self._parse_annotation(item)
            record = participant_data[participant]
            frame_name = item["frame"]
            record["names"].append(frame_name)
            record["face"][frame_name] = face_box
            record["body"][frame_name] = body_box
            record["relation"][frame_name] = relation

        self.name_A, self.video_name_A = self._group_participant(participant_data["A"])
        self.name_B, self.video_name_B = self._group_participant(participant_data["B"])
        self.alignedName_A, self.alignedName_B = sync_dictionaries(
            self.name_A,
            self.name_B,
        )
        self.alignedInfo_A = align_info(self.alignedName_A, self.data_name)
        self.alignedInfo_B = align_info(self.alignedName_B, self.data_name)

        self.keys_A = list(self.alignedInfo_A)
        self.keys_B = list(self.alignedInfo_B)
        self.num_video_parts = _video_part_count(self.data_name)
        self.sample_video_ids = [
            "_".join(key.split("_")[: self.num_video_parts])
            for key in self.keys_A
        ]
        self.sample_labels = [
            _normalise_label(self.alignedInfo_A[key]["relation_classes"][0])
            for key in self.keys_A
        ]

    @staticmethod
    def _load_pickle(path):
        with open(path, "rb") as file:
            return pkl.load(file, encoding="latin1")

    def _parse_annotation(self, item):
        frame_parts = item["frame"].split("_")
        if "relationship" in item:
            role_id = frame_parts[3]
            face_box = item["face_box"]
            body_box = item["body_box"]
            relation = item["relationship"]
        else:
            role_index = 1 if self.data_name == "NoXi" else 2
            role_id = frame_parts[role_index]
            face_box = item["face"][0]
            body_box = item["body"]
            relation = item["relation"]

        participant = "A" if role_id == "0" else "B"
        return participant, face_box, body_box, relation

    def _group_participant(self, participant):
        return group_boxes_by_category(
            participant["names"],
            participant["face"],
            participant["body"],
            participant["relation"],
            self.data_name,
        )

    def __len__(self):
        return len(self.keys_A)

    def __getitem__(self, idx):
        key_a = self.keys_A[idx]
        key_b = self.keys_B[idx]
        position = int(key_a.split("_")[-1])
        video_length = self.video_name_A[self.sample_video_ids[idx]]

        face_a_roi, body_a_roi, relation_a = clip_level_data(
            self.alignedInfo_A[key_a],
            self.transform,
            self.image_dir,
            self.total_frames,
        )
        face_b_roi, body_b_roi, relation_b = clip_level_data(
            self.alignedInfo_B[key_b],
            self.transform,
            self.image_dir,
            self.total_frames,
        )

        return (
            self.facial_feature[key_a],
            face_a_roi,
            self.body_feature[key_a],
            body_a_roi,
            self.audio_feature[key_a],
            self.text[key_a]["text"],
            relation_a,
            self.facial_feature[key_b],
            face_b_roi,
            self.body_feature[key_b],
            body_b_roi,
            self.audio_feature[key_b],
            self.text[key_b]["text"],
            relation_b,
            position,
            video_length,
        )

