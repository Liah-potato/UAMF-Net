# UAMF-Net

UAMF-Net uses pathology, CT, and gastroscopy images to classify samples as tubular carcinoma, poorly cohesive carcinoma, or benign. It averages predictions from five views of a pathology image and uses their predictive entropy to decide when to consult the available CT and gastroscopy models. Confidence rules then determine whether to revise the pathology prediction.

## Quick start

Run the decision demo with Python 3.10 or later:

```bash
python -m uamf_net demo
```

The demo uses example probabilities to show how the entropy gate and decision rules work. For image training or evaluation, install a PyTorch build for your CPU or CUDA environment, then install the repository requirements:

```bash
python -m pip install -r requirements.txt
```

## Prepare images and weights

Create a local CSV using the [example manifest](examples/manifest.example.csv). Each row represents one pathology image and lists its patient ID, split, class label, and paths to paired images. Keep each patient within one split. Leave `ct` or `gastro` empty when that modality is unavailable. The CT model takes 2D slice images.

```bash
python -m uamf_net validate-manifest \
  --manifest data/manifest.csv --data-root data/images
```

The [reference configuration](configs/reference.json) uses ConvNeXt-Base for pathology, MaxViT-Tiny for CT, and FastViT-MA36 for gastroscopy. Labels `0`, `1`, and `2` correspond to tubular carcinoma, poorly cohesive carcinoma, and benign, respectively. Supply your own images and trained checkpoints for image evaluation.

To import existing three-class weights, run the following command for each modality you plan to use, changing `--modality` and `--source` to match:

```bash
python -m uamf_net import-weights --modality pathology \
  --source weights/pathology.pth --checkpoint-dir checkpoints/run_001
```

To train an expert from class folders, use `train-expert` with the modality's training and validation directories. For example:

```bash
python -m uamf_net train-expert --modality ct \
  --train-dir data/ct/train --selection-dir data/ct/val \
  --class-folders 0 1 2 --checkpoint-dir checkpoints/run_001
```

Use `train --help` for the manifest-based trainer and `train-expert --help` for the modality-specific trainer.

## Evaluate

Run evaluation with the reference thresholds:

```bash
python -m uamf_net evaluate \
  --manifest data/manifest.csv --data-root data/images \
  --checkpoint-dir checkpoints/run_001 --split test --output outputs/test
```

Add `--batch-pathology-views` to process the five pathology views in one model forward per image. Evaluation writes image-level metrics to `metrics.json`, per-image decisions to `predictions.jsonl`, and the run settings to `config.json`.

### Select thresholds on validation data

For a new experiment, collect validation predictions and use `tune` to save the selected thresholds:

```bash
python -m uamf_net evaluate \
  --manifest data/manifest.csv --data-root data/images \
  --checkpoint-dir checkpoints/run_001 --split val --output outputs/val

python -m uamf_net tune \
  --predictions outputs/val/predictions.jsonl \
  --output-config configs/local_selected.json \
  --report outputs/val/threshold_search.json
```

Pass `--config configs/local_selected.json` to the test evaluation command. Use the same pathology-view batching mode for validation and test.

## Method and source

- [Decision rules and equations](docs/METHOD.md)
- [Image models and view transforms](uamf_net/models.py)
- [Entropy gate and arbitration](uamf_net/fusion.py)
- [Evaluation and output files](uamf_net/pipeline.py)

Run the repository tests with `python -m unittest discover -s tests -v`.
