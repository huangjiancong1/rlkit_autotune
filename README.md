# Autotune

## 1. Installation
Install https://github.com/vitchyr/multiworld and `pip install -e .` (Similar with https://github.com/vitchyr/rlkit).
Pytorch version: `1.1.0` with or without gpu is ok.

Create your personal `rlkit/launchers/config.py` like: https://drive.google.com/file/d/1Nl-olFC-h5MRG91QoDirEoZb0iF5AmYj/view?usp=sharing.



## 2. The Simulated Experiment
```
python examples/autotune_all/rig_sac_push_autotune.py
python examples/autotune_all/debug_autotune_r_size.py
```

## 3. The Hardware Experiment
```
git fetch original ros:ros
git checkout ros
python examples/rig_sac_automatic/elfin_reach_auto.py
```

## References
The algorithm of Auto-tuning is based on the following paper:

[Hyperparameter Auto-tuning in Self-Supervised Robotic Learning](https://arxiv.org/abs/2010.08252). Jiancong Huang, Juan Rojas, Matthieu Zimmer, Hongmin Wu, Yisheng Guan, and Paul Weng. arXiv preprint, 2021.

