while [ ! -f /content/drive/MyDrive/SEARLE/outputs/eval2.done ]; do sleep 20; done
cd /content/SEARLE
/content/venv/bin/python src/validate.py --exp-name phi_B32 --eval-type phi --dataset fashioniq --dataset-path /content/drive/MyDrive/SEARLE/datasets/FashionIQ --phi-checkpoint-name phi_last.pt > /content/drive/MyDrive/SEARLE/outputs/logs/eval3_phi_fashioniq.log 2>&1
touch /content/drive/MyDrive/SEARLE/outputs/eval3.done
