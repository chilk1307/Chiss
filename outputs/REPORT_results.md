# SEARLE reproduction - evaluation results (val splits, ViT-B/32)

phi_B32 = our reproduction (OTI -> train_phi, paper hyperparameters). SEARLE = official pretrained B/32 baseline.

## CIRR (val)
| | R@1 | R@5 | R@10 | R@50 | Group R@1 | R@2 | R@3 |
|---|---|---|---|---|---|---|---|
| phi_B32 (ours) | 25.47 | 55.23 | 69.24 | 90.43 | 55.20 | 76.11 | 87.83 |
| SEARLE official | 25.66 | 55.47 | 69.24 | 90.58 | 55.08 | 76.58 | 87.85 |

## FashionIQ (val, R@10 / R@50) - see deviation note below
| | Dress | Shirt | Toptee | Average |
|---|---|---|---|---|
| phi_B32 (ours) | 19.08 / 38.53 | 41.84 / 63.18 | 48.61 / 69.44 | 36.51 / 57.05 |
| SEARLE official | 19.67 / 38.53 | 42.68 / 65.27 | 48.61 / 66.67 | 36.99 / 56.82 |

## CIRCO (val)
| | mAP@5 | @10 | @25 | @50 | Recall@5 | @10 | @25 | @50 |
|---|---|---|---|---|---|---|---|---|
| phi_B32 (ours) | 6.42 | 7.15 | 8.48 | 9.08 | 17.73 | 23.18 | 35.45 | 47.27 |
| SEARLE official | 6.82 | 7.78 | 9.14 | 9.76 | 15.91 | 23.18 | 36.36 | 47.27 |

## Deviations / notes
- OTI was run in 3 shards of sorted ImageNet test filenames (seeds 42/43/44), merged into oti_B32 (100,000 images).
- FashionIQ: about 2,431 images are dead links and were skipped (existence filter in datasets.py: triplets with a missing candidate/target image and image names without a file are dropped, filter applied before duplicate removal; plus a 6x retry wrapper around PIL.Image.open). Numbers are NOT strictly comparable to the paper. Patched file: outputs/scripts/datasets_patched.py.
- Comet disabled (no API key). setuptools 69.5.1 needed for comet_ml (pkg_resources).
- Upstream bug: train_metrics.csv keeps only the last epoch; full per-epoch loss in phi_models/phi_B32/train_metrics_full.csv (3-decimal precision from logs).
- Colab VM was reset once after training; environment rebuilt from pip_freeze.txt.
