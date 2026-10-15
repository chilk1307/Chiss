set -x
L=/content/drive/MyDrive/SEARLE/outputs/logs
PY=/content/venv/bin/python
mkdir -p /content/data
tar -xf /content/drive/MyDrive/SEARLE/datasets/FashionIQ.tar -C /content/data
ls /content/data
FD=$(dirname $(find /content/data -maxdepth 3 -type d -name image_splits | head -1))
echo $FD
cd /content/SEARLE
$PY src/validate.py --exp-name phi_B32 --eval-type phi --dataset fashioniq --dataset-path $FD --phi-checkpoint-name phi_last.pt > $L/eval4_phi_fashioniq.log 2>&1
$PY src/validate.py --eval-type searle --dataset fashioniq --dataset-path $FD > $L/eval4_searle_fashioniq.log 2>&1
cp /content/run_fiq4.sh /content/run_resume.sh /content/drive/MyDrive/SEARLE/outputs/scripts/
touch /content/drive/MyDrive/SEARLE/outputs/eval4_fiq.done
