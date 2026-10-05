## 🔥 Model Training

### Setup

The code was developed and tested with Ubuntu 20.04, Python 3.10, PyTorch 2.2.2, and CUDA 12.1.

```bash
pip install -r requirements.txt
```

### Data Preparation

The released ETH/UCY data package should be extracted under `datasets/`. Preprocessed data can be used directly. To preprocess the raw data, place the ETH/UCY resources under `datasets/ETH-UCY/` with the required `annotation/`, `homography/`, `reference/`, and `segmentation/` directories, then run:

```bash
python -m utils.preprocess_dataset --model_config <path_to_model_config>

# Example: ETH
python -m utils.preprocess_dataset --model_config ./configs/model/SAFE_eth.yaml
```

The provided configurations follow the standard leave-one-out protocol:

| Test scene | Configuration |
| :-- | :-- |
| ETH | `configs/model/SAFE_eth.yaml` |
| HOTEL | `configs/model/SAFE_hotel.yaml` |
| UNIV | `configs/model/SAFE_univ.yaml` |
| ZARA1 | `configs/model/SAFE_zara1.yaml` |
| ZARA2 | `configs/model/SAFE_zara2.yaml` |

Training the scene-layout predictor requires the SegFormer-B0 initialization checkpoint at:

```text
checkpoints/pretrained/segformer-b0-finetuned-cityscapes-1024-1024/
```

The checkpoint can be obtained from [Hugging Face](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024) or from the released project assets.

### Train SAFE

SAFE consists of a scene-layout predictor, a scene-anchored flow emitter, and an interaction-aware rollout simulator. Train all three components with the same scene configuration.

```bash
CONFIG=./configs/model/SAFE_eth.yaml

# 1. Train the scene-layout predictor for appearance and population maps.
python trainval.py --model_train emitter_pre --model_config "${CONFIG}"

# 2. Train the scene-anchored flow emitter.
python trainval.py --model_train emitter --model_config "${CONFIG}"

# 3. Train the interaction-aware rollout simulator.
python trainval.py --model_train simulator --model_config "${CONFIG}"
```

Checkpoints are saved under `checkpoints/<scene>/`. All three checkpoints are required for evaluation and inference.

<br>

## 📊 Model Evaluation

### Pretrained Models

Download the released checkpoint archive and extract it under `checkpoints/` while preserving the directory structure.

### Evaluate SAFE

```bash
python trainval.py --test --model_config <path_to_model_config>

# Example: ETH
python trainval.py --test --model_config ./configs/model/SAFE_eth.yaml
```

Evaluation performs 20 complete rollout trials for each test scene. It saves generated trajectories under `output/generated/<scene>/` and aggregate metrics under `output/log/<scene>/`.

<br>

## 🚀 Model Inference

### Export Generated Trajectories

```bash
python trainval.py --export --model_config <path_to_model_config>

# Example: ETH
python trainval.py --export --model_config ./configs/model/SAFE_eth.yaml
```

By default, this command generates one 10-minute crowd scenario per test scene. The outputs include trajectory CSV files, Social-GAN-format trajectory files, predicted appearance and population maps, and visualization videos under `output/generated/<scene>/`.

The export duration, number of trials, and output formats can be configured in `SAFM/evaluate_export_generated_traj.py`.

### Generate on a Custom Synthetic Scene

Synthetic inference requires a complete scene bundle rather than a single image. For a scene named `<scene>`, provide:

```text
datasets/Synthetic/
├── image_terrain/<scene>_bg.png
├── gt/<scene>_appearance_density.png
├── gt/<scene>_population_density.png
└── homography/<scene>_scale.txt
```

Set `SCENE_LIST` in `SAFM/evaluate_synthetic_dataset.py`, then run:

```bash
python trainval.py --synthetic --model_config <path_to_model_config>

# Example
python trainval.py --synthetic --model_config ./configs/model/SAFE_eth.yaml
```

Generated trajectories and optional visualization videos are written to `output/generated/synthetic/`. Navigation meshes are cached automatically in `datasets/Synthetic/navmesh/`.
