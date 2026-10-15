D=/content/drive/MyDrive/SEARLE/datasets
L=/content/drive/MyDrive/SEARLE/outputs/logs
PY=/content/venv/bin/python
cd /content/SEARLE
$PY src/validate.py --exp-name phi_B32 --eval-type phi --dataset fashioniq --dataset-path $D/FashionIQ --phi-checkpoint-name phi_last.pt > $L/eval5_phi_fashioniq.log 2>&1
$PY src/validate.py --eval-type searle --dataset fashioniq --dataset-path $D/FashionIQ > $L/eval5_searle_fashioniq.log 2>&1
touch /content/drive/MyDrive/SEARLE/outputs/eval5_fiq.done
