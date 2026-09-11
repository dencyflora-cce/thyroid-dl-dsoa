# Hybrid Deep Learning Approach for Thyroid Disease Detection

Reference implementation accompanying the manuscript **“Hybrid Deep Learning Approach for Thyroid Disease Detection using EfficientNet-B7 and Capsule Network.”**

**Authors:** Dency Flora G and Venkataramanan C  
**Affiliation:** Sri Eshwar College of Engineering, Coimbatore, India

## Scope and reproducibility statement

This repository provides a clean reference implementation reconstructed from the methods and experimental settings documented in the manuscript. The original experimental source code was not available to the authors for release. Therefore, this repository must **not** be represented as the exact historical source code that generated the manuscript's reported numerical results unless the authors independently verify that equivalence.

The implementation covers the principal computational components described in the manuscript: image preprocessing and Min–Max normalization, EfficientNet-B7 transfer learning/partial fine-tuning, a Capsule Network classifier with dynamic routing, and the Domestic Sheep Optimization Algorithm (DSOA) used for hyperparameter search.

Some details required for exact historical reproduction are not fully specified in the manuscript, including complete search bounds for batch size and EfficientNet fine-tuning parameters and the fully rendered mathematical form of the DSOA update equation. Those details are therefore exposed as configurable reference choices rather than presented as undocumented historical facts.

## Dataset

The study uses a single publicly available third-party Kaggle repository:

https://www.kaggle.com/datasets/shreeyuva25/thyroid-dataset

The manuscript reports 1,607 ultrasound images: 632 benign, 804 malignant, and 171 normal. The dataset is **not redistributed** in this repository.

Expected local layout:

```text
data/raw/
├── benign/
├── malignant/
└── normal/
```

The manuscript states that patient-level identifiers were unavailable and that the data were divided at the image level into training, validation and testing sets using an 80:10:10 ratio. Augmentation was applied to the training data only.

## Experimental settings documented in the manuscript

| Setting | Value |
|---|---:|
| Input image size | 224 × 224 |
| Normalization | Min–Max |
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Capsule vector dimension | 16 |
| Routing iterations | 3 |
| Batch size | 16 |
| Maximum epochs | 50 |
| Dropout | 0.5 |
| DSOA population | 30 |
| DSOA iterations | 100 |
| Learning-rate search | 0.0001–0.01 |
| Capsule-dimension search | 8–32 |
| Routing-iteration search | 2–5 |
| Exploration coefficient α | 0.5 |
| Exploitation coefficient β | 1.5 |
| Boundary handling | Clipping to valid bounds |
| Random seed | 42 |

The manuscript also reports Python 3.9, TensorFlow/Keras 2.11.0, NumPy 1.23.5, Pandas 1.5.3, OpenCV 4.7.0, Ubuntu 20.04 LTS, CUDA 11.8 and cuDNN 8.6.

## Installation

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the pipeline

Train the model:

```bash
python main.py train
```

Evaluate the saved model on the held-out test split:

```bash
python main.py evaluate
```

Run a small, self-contained DSOA software smoke test (does not train the neural network):

```bash
python main.py dsoa-smoke
```

The DSOA class also accepts any user-supplied validation objective. A complete historical DSOA search cannot be reconstructed solely from the manuscript because some search dimensions/bounds and the exact rendered update equation are unavailable in the released manuscript text.

## Outputs

Training creates:

```text
outputs/training_log.csv
outputs/split_sizes.json
checkpoints/best_model.keras
checkpoints/final_model.keras
```

Evaluation creates:

```text
outputs/test_metrics.json
```

## Reported manuscript results

The manuscript reports mean performance of 98.17% accuracy, 98.36% precision, 98.11% recall, 98.23% F1-score and 98.42% specificity. These values are reported manuscript results and are **not presented here as independently reproduced results** from this reference implementation.

## Ethical/data provenance note

The authors did not collect the original ultrasound images or clinical information and did not have access to the original patient-level data. The public repository did not provide the authors with patient identifiers, clinical records, or additional pathological/cytological information. Information concerning the original clinical data-collection procedures, ethics approval and informed consent was not available to the authors through the released dataset materials. No unsupported claims about the original clinical ethics process are made by this repository.

## License

See `LICENSE.txt`.
