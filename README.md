# [TSMC] Ask-AC: An Initiative Advisor-in-the-Loop Actor-Critic Framework

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2207.01955-b31b1b.svg)](https://arxiv.org/abs/2207.01955)

Official codebase for paper [Ask-AC: An Initiative Advisor-in-the-Loop Actor-Critic Framework](https://arxiv.org/abs/2207.01955). This codebase is based on the open-source [stable-baseline3](https://github.com/DLR-RM/stable-baselines3) framework and please refer to that repo for more documentation.

<div align="center">
<img src="https://github.com/liushunyu/Ask-AC/blob/master/introduction.png" width="50%">
</div>

## Overview

**TLDR:** Our contribution is a dedicated initiative advisor-in-the-loop actor-critic framework for interactive reinforcement learning, which enables a two-way message passing and seeks advisor assistance only on demand. The proposed Ask-AC substantially lessens the advisor participation effort and is readily applicable to various discrete actor-critic architectures. 

**Abstract:** Despite the promising results achieved, state-of-the-art interactive reinforcement learning schemes rely on passively receiving supervision signals from advisor experts, in the form of either continuous monitoring or pre-defined rules, which inevitably result in a cumbersome and expensive learning process. In this paper, we introduce a novel initiative advisor-in-the-loop actor-critic framework, termed as Ask-AC, that replaces the unilateral advisor-guidance mechanism with a bidirectional learner-initiative one, and thereby enables a customized and efficacious message exchange between learner and advisor. At the heart of Ask-AC are two complementary components, namely action requester and adaptive state selector, that can be readily incorporated into various discrete actor-critic architectures. The former component allows the agent to initiatively seek advisor intervention in the presence of uncertain states, while the latter identifies the unstable states potentially missed by the former especially when environment changes, and then learns to promote the ask action on such states. Experimental results on both stationary and non-stationary environments and across different actor-critic backbones demonstrate that the proposed framework significantly improves the learning efficiency of the agent, and achieves the performances on par with those obtained by continuous advisor monitoring.


![image](https://github.com/liushunyu/Ask-AC/blob/master/framework.png)



## Prerequisites

#### Install dependencies

See `requirments.txt` file for more information about how to install the dependencies.


## Usage

Please follow the instructions below to replicate the results in the paper.

```bash
# test the game-playing performance of the human advisor
python test_human.py
```

![image](https://github.com/liushunyu/Ask-AC/blob/master/test-human.png)

```bash
# train the agent with the human advisor
python train_human.py --exp run
```

![image](https://github.com/liushunyu/Ask-AC/blob/master/train-human.png)


## Citation

If you find this work useful for your research, please cite our paper:

```
@article{liu2022AskAC,
  title={Ask-AC: An Initiative Advisor-in-the-Loop Actor-Critic Framework},
  author={Liu, Shunyu and Wang, Xinchao and Yu, Na and Song, Jie and Chen, Kaixuan and Feng, Zunlei and Song, Mingli},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TSMC.2023.3296773}
}
```

## Contact

Please feel free to contact me via email (<liushunyu@zju.edu.cn>) if you are interested in my research :)
