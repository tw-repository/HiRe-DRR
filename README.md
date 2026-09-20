# HiRe-DRR
**Hierarchical Multimodal Interaction Reasoning for Dyadic Relationship Recognition**
HiRe-DRR is a multimodal framework for recognising social relationships from dyadic interactions. It jointly models low- and high-level cues from facial behaviour, body motion, audio and speech transcripts.

The framework constructs homogeneous and heterogeneous interactions between two participants and dynamically selects informative interactions through the Mixture-of-Interaction Transformer (MoI-Former). Hierarchy-aware conditional supervision is used to preserve the progressive distinctions between relationship categories.

## Highlights

- Joint modelling of face, body, audio and text modalities.
- Separate representations for low- and high-level behavioural cues.
- Homogeneous and heterogeneous cross-participant interaction modelling.
- Relation-aware Top-K interaction routing.
- Hierarchy-aware conditional loss for relationship recognition.
- Support for NoXi, UDIVA and Seamless Interaction.

## Environment
Install the required dependencies using:

```bash
pip install -r environment.txt
```

## Dataset
The datasets are distributed by their respective owners. Please follow the official access procedures and licence requirements.

### NoXi

[NoXi](https://multimediate-challenge.org/datasets/Dataset_NoXi/) is a multimodal corpus of mediated novice–expert interactions introduced by [Cafaro et al., ICMI 2017](https://doi.org/10.1145/3136755.3136780).

We use four relationship categories:

- Stranger
- Acquaintance
- Friend
- Very Good Friend


### UDIVA

[UDIVA](https://chalearnlap.cvc.uab.cat/dataset/41/description/) is a multimodal dataset of face-to-face dyadic interactions introduced by [Palmero et al., WACV Workshops 2021](https://openaccess.thecvf.com/content/WACV2021W/HBU/html/Palmero_Context-Aware_Personality_Inference_in_Dyadic_Scenarios_Introducing_the_UDIVA_Dataset_WACVW_2021_paper.html).

We use two relationship categories:

- Known
- Unknown

### Seamless Interaction

[Seamless Interaction](https://github.com/facebookresearch/seamless_interaction) is a large-scale audiovisual dataset of face-to-face dyadic interactions introduced by [Agrawal et al. 2025](https://ai.meta.com/research/publications/seamless-interaction-dyadic-audiovisual-motion-modeling-and-large-scale-dataset/).

We use five relationship categories:

- Stranger
- Acquaintance
- Friend
- Family
- Couple

## Low- and High-Level Feature Extraction

| Modality | Low-level features | Extractor | High-level features | Extractor |
|:---:|:---|:---:|:---|:---:|
| **Face** | Facial landmarks<br>Head-pose coordinates | [OpenFace 2.0](https://github.com/TadasBaltrusaitis/OpenFace) | Facial action units<br>Eye-gaze directions<br>Valence and arousal<br>Visual appearance | [OpenFace 2.0](https://github.com/TadasBaltrusaitis/OpenFace)<br>[EmoNet](https://github.com/face-analysis/emonet)<br>[TimeSformer](https://github.com/facebookresearch/TimeSformer) |
| **Body** | Body-pose coordinates | [Keypoint R-CNN](https://docs.pytorch.org/vision/stable/models/keypoint_rcnn.html) | Body openness and orientation<br>Visual appearance | Pose-based geometric modelling<br>[TimeSformer](https://github.com/facebookresearch/TimeSformer) |
| **Audio** | 20 MFCCs<br>Fundamental frequency (F0)<br>Root mean square energy | [Librosa](https://github.com/librosa/librosa) | Acoustic emotion<br>Spectral flux<br>Speech and silence duration<br>Pause duration and frequency<br>Alpha ratio<br>Hammarberg index | [HuBERT-Large](https://huggingface.co/superb/hubert-large-superb-er)<br>[Librosa](https://github.com/librosa/librosa)<br>[openSMILE](https://github.com/audeering/opensmile) |
| **Text** | Semantic embedding | [BERT-base](https://huggingface.co/google-bert/bert-base-uncased) | Utterance-level sentiment representation | [RoBERTa-base](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) |

The training code expects facial, body and audio descriptors to have been extracted and saved as pickle files. BERT and Twitter-RoBERTa representations are computed from the stored transcripts during model inference.

## Data Preparation

Each dataset requires the following inputs:

- A directory containing the sampled video frames.
- Training and testing annotation files.
- Pre-extracted facial feature files.
- Pre-extracted body feature files.
- Pre-extracted audio feature files.
- Speech transcript files.
- A Seamless Interaction identity-projection file when using `SeamInt`.

Dataset-specific paths can be provided through command-line arguments. The pickle dictionaries for the two participants must use aligned clip identifiers.

## Command-Line Arguments

### Training Configuration

| Argument | Default | Description |
|:---|:---:|:---|
| `--mode` | `NoXi` | Dataset: `NoXi`, `UDIVA` or `SeamInt`. |
| `--person` | `B` | Target participant for NoXi: `A` or `B`. |
| `--k-fold` | `3` | Cross-validation fold identifier. |
| `-b`, `--batch-size` | `64` | Training and evaluation batch size. |
| `-j`, `--workers` | `4` | Number of data-loading workers. |
| `--epochs` | `100` | Number of training epochs. |
| `--lr` | `0.0001` | Initial learning rate. |
| `--lr-step` | `15` | Number of epochs between learning-rate reductions. |
| `--start-epoch` | `1` | Initial epoch index. |
| `--total-frames` | `10` | Number of uniformly sampled frames in each clip. |
| `--seed` | `42` | Random seed. |
| `--device` | `cuda` or `cpu` | Device used for training and evaluation. |
| `--analyse-attention` | Disabled | Print low/high-level routing preferences and Top-K interactions. |

### Dataset Paths

| Argument | Description |
|:---|:---|
| `--data-dir` | Directory containing sampled video frames. |
| `--facial-feature-dir` | Pickle file containing facial features. |
| `--body-feature-dir` | Pickle file containing body features. |
| `--audio-feature-dir` | Pickle file containing audio features. |
| `--text-dir` | Pickle file containing speech transcripts. |
| `--train-list` | Training annotation file. |
| `--test-list` | Testing annotation file. |
| `--save-dir` | Directory used to save model checkpoints. |
| `--id-projection-path` | Seamless Interaction participant-ID mapping file. |

### Pre-trained Model Paths

| Argument | Description |
|:---|:---|
| `--timesformer-path` | Path to the pre-trained TimeSformer checkpoint. |
| `--bert-path` | Local path or Hugging Face identifier for BERT. |
| `--sentiment-model-path` | Local path or Hugging Face identifier for the sentiment model. |


## Training

### NoXi

```bash
python train_att.py \
  --mode NoXi \
  --person B \
  --k-fold 1 \
  --batch-size 64 \
  --epochs 100 \
  --lr 0.0001 \
  --total-frames 10 \
  --device cuda:0 \
  --data-dir /path/to/NoXi/frames \
  --facial-feature-dir /path/to/NoXi_face_features.pkl \
  --body-feature-dir /path/to/NoXi_body_features.pkl \
  --audio-feature-dir /path/to/NoXi_audio_features.pkl \
  --text-dir /path/to/NoXi_transcripts.pkl \
  --train-list /path/to/NoXi_train_bbox1.pkl \
  --test-list /path/to/NoXi_test_bbox1.pkl \
  --save-dir /path/to/checkpoints/NoXi \
  --timesformer-path /path/to/timesformer_checkpoint.pyth \
  --bert-path google-bert/bert-base-uncased \
  --sentiment-model-path cardiffnlp/twitter-roberta-base-sentiment-latest
```

### UDIVA

```bash
python train_att.py \
  --mode UDIVA \
  --k-fold 1 \
  --batch-size 64 \
  --device cuda:0 \
  --data-dir /path/to/UDIVA/frames \
  --facial-feature-dir /path/to/UDIVA_face_features.pkl \
  --body-feature-dir /path/to/UDIVA_body_features.pkl \
  --audio-feature-dir /path/to/UDIVA_audio_features.pkl \
  --text-dir /path/to/UDIVA_transcripts.pkl \
  --train-list /path/to/UDIVA_train_bbox1.pkl \
  --test-list /path/to/UDIVA_test_bbox1.pkl \
  --save-dir /path/to/checkpoints/UDIVA \
  --timesformer-path /path/to/timesformer_checkpoint.pyth
```

### Seamless Interaction

```bash
python train_att.py \
  --mode SeamInt \
  --k-fold 1 \
  --batch-size 64 \
  --device cuda:0 \
  --data-dir /path/to/SeamInt/frames \
  --facial-feature-dir /path/to/SeamInt_face_features.pkl \
  --body-feature-dir /path/to/SeamInt_body_features.pkl \
  --audio-feature-dir /path/to/SeamInt_audio_features.pkl \
  --text-dir /path/to/SeamInt_transcripts.pkl \
  --train-list /path/to/SeamInt_train_bbox1.pkl \
  --test-list /path/to/SeamInt_test_bbox1.pkl \
  --id-projection-path /path/to/id_proj.pkl \
  --save-dir /path/to/checkpoints/SeamInt \
  --timesformer-path /path/to/timesformer_checkpoint.pyth
```

## Evaluation

Validation is performed after every training epoch. The implementation reports:

- Class-wise recall
- Unweighted average recall (UAR)
- Macro F1 score
- Mean average precision (mAP)

A checkpoint is saved whenever either UAR or mAP improves.


## Contributing

For questions, please open a GitHub issue or contact:

**Tang Wang** — [wangtang@cuit.edu.cn](mailto:wangtang@cuit.edu.cn)
