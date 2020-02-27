# Deep Multi-attributed Graph Translation with Node-Edge Co-evolution.
![image_text](images/NEC-DGT.png)
This repository is the official Tensorflow implementation of NEC-DGT, a multi-attributed graph translation model.

The relevant paper is ["Deep Multi-attributed Graph Translation with Node-Edge Co-evolution"](http://mason.gmu.edu/~lzhao9/materials/papers/ICDM_2019_NEC_DGT-final.pdf).

[Xiaojie Guo](https://sites.google.com/view/xiaojie-guo-personal-site), [Liang Zhao](http://mason.gmu.edu/~lzhao9/), Cameron Nowzari, Setareh Rafatirad, Houman Homayoun, and Sai Dinakarrao (ICDM 2019 Best Paper Award).

## Installation
Install Tensorflow following the instuctions on the official website. The code has been tested over Tensorflow 1.13.1 version.


## Code Description

There are four experiment tasks: synthetic1, synthetic2, Molecule (reactaction prediction) and IoT malware confinement.
For each experiment task: `main.py` is the main executable file which includes specific arguments and training iterations and calls `model.py` and `utlis.py`. `utlis.py` is where the dataset is read.


## Run the code
For each task, to train a model,edit the FLAGS "--type" to "train" and:

               python main.py
               
               
to test a model,edit the FLAGS "--type" to "test" and:

               python main.py             

