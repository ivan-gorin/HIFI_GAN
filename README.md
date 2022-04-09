# HIFI GAN

Implementation of HIFI GAN network: [original paper](https://arxiv.org/abs/2010.05646)

Tested on python 3.7.12

Setup instructions:
```
pip install -r requirements.txt
chmod +x setup.sh
./setup.sh
```

Train with
```
python train.py --config config.json
```

Test with
```
python test.py --config config.json --model PATH_TO_CHECKPOINT
```

[Report](https://wandb.ai/ivan-gorin/HIFI/reports/HIFI-GAN-Report--VmlldzoxMzU4MjU2) (in Russian)
