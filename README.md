# Autotune

## 1. Installation
Install https://github.com/vitchyr/multiworld and `pip install -e .` (Similar with https://github.com/vitchyr/rlkit)

Create your personal `rlkit/launchers/config.py` like: https://drive.google.com/file/d/1Nl-olFC-h5MRG91QoDirEoZb0iF5AmYj/view?usp=sharing

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

## 4. Other
Video about how the diversity of the hardware environment change online: https://drive.google.com/file/d/1GLYQ4bP7t_oQQhuy84qsk1WMIcPAomP0/view?usp=sharing
.
