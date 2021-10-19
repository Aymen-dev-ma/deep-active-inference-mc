# Deep active inference agents using Monte-Carlo methods

This source code release accompanies the manuscript:

Z. Fountas, N. Sajid, P. A.M. Mediano and K. Friston "[Deep active inference agents using Monte-Carlo methods](https://papers.nips.cc/paper/2020/hash/865dfbde8a344b44095495f3591f7407-Abstract.html)",	Advances in Neural Information Processing Systems 33 (NeurIPS 2020).

If you use this model or the dynamic dSprites environment in your work, please cite our paper.

---
### Description
For a quick overview see this [video](https://www.youtube.com/watch?v=KA-lZ3krVtU). In this work, we propose the deep neural architecture illustrated below, which can be used to train scaled-up active inference agents for continuous complex environments based on amortized inference, M-C tree search, M-C dropouts and top-down transition precision, that encourages disentangled latent representations.

<p align="center"><img src="architecture.png" width="700"></p>

We test this architecture on two tasks from the Animal-AI Olympics and a new simple object-sorting task based on [DeepMind's dSprites dataset](https://github.com/deepmind/dsprites-dataset/).

### Demo behavior

<table style="width:100%;">
  <tr>
    <td align="center"><img src="dsprites.gif" width="200" height="200"/></td>
    <td align="center"><img src="animalai.gif" width="200" height="200"/></td>
  </tr>
  <tr>
    <td align="center">Agent trained in the Dynamic dSprites environment</td>
    <td align="center">Agent trained in the Animal-AI environment</td>
  </tr>
</table>

### Requirements
* Programming language: Python 3
* Libraries: tensorflow >= 2.0.0, numpy, matplotlib, scipy, opencv-python
* [dSprites dataset](https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz).

### Instructions

##### Installation

* Initially, make sure the required libraries are installed in your computer. Open a terminal and type
```bash
pip install -r requirements.txt
```

* Then, clone this repository, navigate to the project directory and download the dSrpites dataset by typing
```bash
wget https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```
or by manually visiting the above URL.

##### Training
* To train an active inference agent to solve the dynamic dSprites task, type
```bash
python train.py
```
This script will automatically generate checkpoints with the optimized parameters of the agent and store this checkpoints to a different sub-folder every 25 training iterations. The default folder that will contain all sub-folders is ```figs_final_model_0.01_30_1.0_50_10_5```. The script will also generate a number of performance figures, also stored in the same folder. You can stop the process at any point by pressing ```Ctr+c```.

##### Testing
* Finally, once training has been completed, the performance of the newly-trained agent can be demonstrated in real-time by typing
```bash
python test_demo.py -n figs_final_model_0.01_30_1.0_50_10_5/checkpoints/ -m
```
This command will open a graphical interface which can be controlled by a number of keyboard shortcuts. In particular, press:

  * `q` or `esc` to exit the simulation at any point.
  * `1` to enable the MCTS-based full-scale active inference agent (enable by default).
  * `2` to enable the active inference agent that minimizes expected free energy calculated only for a single time-step into the future.
  * `3` to make the agent being controlled entirely by the habitual network (see manuscript for explanation)
  * `4` to activate *manual mode* where the agents are disabled and the environment can be manipulated by the user. Use the keys `w`, `s`, `a` or `d` to move the current object up, down, left or right respectively.
  * `5` to enable an agent that minimizes the terms `a` and `b` of equation 8 in the manuscript.
  * `6` to enable an agent that minimizes only the term `a` of the same equation (reward-seeking agent).
  * `m` to toggle the use of sampling in calculating future transitions.


  ### Bibtex
  ```
@inproceedings{fountas2020daimc,
 author = {Fountas, Zafeirios and Sajid, Noor and Mediano, Pedro and Friston, Karl},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {11662--11675},
 publisher = {Curran Associates, Inc.},
 title = {Deep active inference agents using Monte-Carlo methods},
 url = {https://proceedings.neurips.cc/paper/2020/file/865dfbde8a344b44095495f3591f7407-Paper.pdf},
 volume = {33},
 year = {2020}
}
  ```
