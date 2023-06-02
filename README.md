# Ask-AC: An Initiative Advisor-in-the-Loop Actor-Critic Framework

Official codebase for paper [Ask-AC: An Initiative Advisor-in-the-Loop Actor-Critic Framework](https://arxiv.org/abs/2207.01955). This codebase is based on the open-source [stable-baseline3](https://github.com/DLR-RM/stable-baselines3) framework and please refer to that repo for more documentation.



## 1. Prerequisites

#### Install dependencies

See `requirment.txt` file for more information about how to install the dependencies.


## 2. Usage

Please follow the instructions below to replicate the results in the paper.

```bash
# test the game-playing performance of the human advisor
python test_human.py

# train the agent with the human advisor
python train_human.py --exp run
```

## 3. Citation

If you find this work useful for your research, please cite our paper:

```
@article{liu2022AskAC,
  title={Ask-AC: An Initiative Advisor-in-the-Loop Actor-Critic Framework},
  author={Liu, Shunyu and Wang, Xinchao and Yu, Na and Song, Jie and Chen, Kaixuan and Feng, Zunlei and Song, Mingli},
  journal={arXiv preprint arXiv:2207.01955},
  year={2022}
}
```
