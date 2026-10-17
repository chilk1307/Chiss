SEARLE reproduction (ViT-B/32) - status 2026-10-15

PIPELINE: image_concepts_association -> OTI -> train_phi -> validate.py. Repo commit a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37.
SEEDS: OTI 3 shards seeds 42/43/44 (deviation, see below); train_phi seed 42.
OTI: 350 steps, lr 2e-2, batch 32, top-k 15, lambda_gpt 0.5, ema 0.99.
PHI: 100 epochs, batch 256, lr 1e-4, lambda_distil 1, lambda_gpt 0.75, temperature 0.25, dropout 0.5, --save-training. Final loss epoch 100: total 4.711 gpt 0.139 distil 4.607.
ENV: python3.8, torch 1.11.0+cu113, torchvision 0.12.0+cu113, transformers 4.24.0, pandas 1.4.2, numpy 1.23.5, comet-ml 3.33.6 (disabled), CLIP d05afc43, setuptools 69.5.1 (needed for comet_ml import). Full list: outputs/pip_freeze.txt. GPU A100.

RESULTS (validation split)
phi_B32 (this reproduction, phi_last.pt):
 CIRR: R@1 25.47, R@5 55.23, R@10 69.24, R@50 90.43, Group R@1/2/3 55.20/76.11/87.83
  CIRCO: 17.73 / 23.18 / 35.45 / 47.27 (log keys circo_recall_at5/10/25/50 as printed by validate.py)
   FashionIQ: NOT DONE
   Official SEARLE B/32 (torch.hub weights):
    CIRR: R@5 55.47, R@10 69.24, R@50 90.58, Group R@1/2/3 55.08/76.58/87.85 (R@1 in logs/eval_searle_cirr.log)
     CIRCO: NOT DONE (run was at ~6%, very slow cold Drive reads)
      FashionIQ: NOT DONE

      DEVIATIONS / NOTES
      1. OTI split into 3 contiguous shards of sorted ImageNet test filenames (seeds 42/43/44), merged into oti_B32 (100,000 unique images). Not a single seed-42 run.
      2. FashionIQ is missing ~2,431 images (dead links). Original code crashes when a whole batch is missing. Prepared patch outputs/scripts/patch_fiq.py filters out samples with missing images in FashionIQDataset.__init__ (only for FashionIQ). Results with it are not directly comparable to the paper.
      3. Comet not used (no API key).
      4. train_metrics.csv has one row only: upstream bug (train_one_epoch concat is not returned). Full per-epoch losses (3 decimals, from the log) are in phi_models/phi_B32/train_metrics_full.csv.
      5. Colab VM reset after training; evaluation environment rebuilt from pip_freeze.txt (outputs/scripts/run_eval2.sh).

      TO RESUME: run outputs/scripts/run_eval2.sh (rebuilds venv, clones repo, runs searle circo, searle fashioniq), apply patch_fiq.py before FashionIQ, then run_fiq3.sh for phi_B32 on FashionIQ. Logs: outputs/logs/eval*_*.log.
