# HiRe-DRR
Hierarchical Multimodal Interaction Reasoning for Dyadic Relationship Recognition

## Environment
Please refer to the "environment.txt".

## Dataset
[NoXi](https://multimediate-challenge.org/datasets/Dataset_NoXi/) was released by [[Angelo et al. ICMI 2017](https://dl.acm.org/doi/abs/10.1145/3136755.3136780?casa_token=8UoDP_iZs3gAAAAA:mYUOHpnezNatC2FQpxXXur2Y8CWiKmS_2Jech1yxEp-XcBU9OsrgC6li0zdN5Up9ornfGimLOv4)]. It involves 4 different relationships: Stranger, Acquaintance, Friend and Very good friend.

[UDIVA](https://chalearnlap.cvc.uab.cat/dataset/41/description/) was released by [[Cristina et al. WACV 2021](https://openaccess.thecvf.com/content/WACV2021W/HBU/html/Palmero_Context-Aware_Personality_Inference_in_Dyadic_Scenarios_Introducing_the_UDIVA_Dataset_WACVW_2021_paper.html)]. It involves 2 relationships: Unknown and Known.

[Seamless Interaction](https://github.com/facebookresearch/seamless_interaction) was released by [[Vasu Agrawal et al. 2025](https://ai.meta.com/research/publications/seamless-interaction-dyadic-audiovisual-motion-modeling-and-large-scale-dataset/)]. 5 relationships used in this study: Stranger, Acquaintance, Friend, Family and Couple.

## Low- and High-level Features Extraction
| Modality | Low-level Features | Extractor | High-level Features | Extractor |
|:---:|:---|:---:|:---|:---:|
| **Face** | Facial landmarks<br>Head pose coordinates | [OpenFace 2.0](https://www.cl.cam.ac.uk/~pr10/publications/wacv16a.pdf) | Facial action units<br>Eye gaze directions<br>Valence and arousal<br>Visual appearance | OpenFace 2.0 [[67]](#references)<br>EmoNet [[77]](#references)<br>TimeSformer [[68]](#references) |
| **Body** | Body pose coordinates | Keypoint R-CNN [[69]](#references) | Body openness and orientation<br>Visual appearance | Pose-based geometric modelling<br>TimeSformer [[68]](#references) |
| **Audio** | 20 MFCCs, F0, RMS | Librosa [[70]](#references) | Acoustic emotion<br>Spectral flux<br>Speech/silence duration<br>Pause duration/frequency<br>Alpha ratio<br>Hammarberg index | HuBERT-Large [[71]](#references)<br>Librosa [[70]](#references)<br>openSMILE [[78]](#references) |
| **Text** | Semantic embedding | BERT-base [[72]](#references) | Sentiment of the speech | DeBERTa-base [[73]](#references) |

## Usage
    optional arguments:
      --mode         Choose NoXi, UDIVA or SeamInt
      --person       For the asymmetric relationship in NoXi, choose either A or B
      --k_fold       K-fold cross validation
      --workers N         number of data loading workers (defult: 4)
      --batch-size N      mini-batch size (default: 1)
      --print-freq N, -p N      print frequency (default: 10)
      -n N, --num-classes N     number of classes
      --total_frames       The number of frames contained in each clip


## Contributing
For any questions, feel free to open an issue or contact us (wangtang@cuit.edu.cn)
