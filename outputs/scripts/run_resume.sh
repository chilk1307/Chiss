set -x
L=/content/drive/MyDrive/SEARLE/outputs/logs
D=/content/drive/MyDrive/SEARLE/datasets
S=/content/drive/MyDrive/SEARLE/outputs/scripts
PY=/content/venv/bin/python
python3 -m pip install -q uv
python3 -m uv venv --python 3.8 /content/venv
python3 -m uv pip install -q --python $PY --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu113 -r /content/drive/MyDrive/SEARLE/outputs/pip_freeze.txt
cd /content
git clone -q https://github.com/miccunifi/SEARLE.git
cd /content/SEARLE
git checkout -q a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37
mkdir -p data
for d in oti_pseudo_tokens similar_concepts GPT_phrases phi_models; do ln -sfn /content/drive/MyDrive/SEARLE/outputs/$d data/$d; done
cp $S/fiq_add.txt /content/fiq_add.txt
$PY $S/patch_fiq.py
$PY -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
$PY src/validate.py --exp-name phi_B32 --eval-type phi --dataset fashioniq --dataset-path $D/FashionIQ --phi-checkpoint-name phi_last.pt > $L/eval3_phi_fashioniq.log 2>&1
$PY src/validate.py --eval-type searle --dataset fashioniq --dataset-path $D/FashionIQ > $L/eval3_searle_fashioniq.log 2>&1
$PY src/validate.py --eval-type searle --dataset circo --dataset-path $D/CIRCO > $L/eval3_searle_circo.log 2>&1
touch /content/drive/MyDrive/SEARLE/outputs/eval3_all.done
