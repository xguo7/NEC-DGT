# Deep Multi-attributed Graph Translation with Node-Edge Co-evolution.
![image_text](images/NEC-DGT.png)
This repository is the official Tensorflow implementation of NEC-DGT, a multi-attributed graph translation model.

The relevant paper is ["Deep Multi-attributed Graph Translation with Node-Edge Co-evolution"](http://mason.gmu.edu/~lzhao9/materials/papers/ICDM_2019_NEC_DGT-final.pdf).
[Xiaojie Guo](https://sites.google.com/view/xiaojie-guo-personal-site), [Liang Zhao](http://mason.gmu.edu/~lzhao9/), Cameron Nowzari, Setareh Rafatirad, Houman Homayoun, and Sai Dinakarrao (ICDM 2019 Best Paper Award).



There are four experiment tasks: synthetic1, synthetic2, Molecule (reactaction prediction) and IoT malware confinement.


Each folder contrains the relevant python script: main.py, model.py and utlis.py, as well as the dataset.


For running each task, run the main.py by editing the FLAGS "--type" to "train" to train the model.

                                       by editing the FLAGS "--type" to "test" to test and use the trained model.
