# On the Convergence of FedAvg on Non-IID Data

This repository contains the codes for the paper

> [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/pdf/1907.02189.pdf)

Our paper is a tentative theoretical understanding towards [FedAvg](<https://arxiv.org/abs/1602.05629>) and how different sampling and averaging schemes affect its convergence.

Our code is based on the codes for [FedProx](<https://github.com/litian96/FedProx>), another federated algorithm used in heterogeneous networks.



## Usage

1. First generate data by the following code. Here `generate_random_niid` is used to generate the dataset named as ` mnist unbalanced ` in our paper,  where the number of samples among devices follows a power law. `generate_equal` is used to generate the dataset named as ` mnist balanced ` where we force all devices to have the same amount of samples. More non-iid distributed datasets could be found in [FedProx](<https://github.com/litian96/FedProx>).

    ```
   cd fedpy
   python data/mnist/generate_random_niid.py
   python data/mnist/generate_equal.py
   python data/synthetic/generate_synthetic.py
   ```

2. Then start to train. You can run a single algorithm on a specific configuration like

    ```
   python main.py --gpu --dataset $DATASET --clients_per_round $K --num_round $T --num_epoch $E --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noaverage --noprint
   ```

#### Notes

- There are three choices for `$ALGO`, namely `fedavg4` (containning the Scheme I and II), `fedavg5` (for the original scheme) and `fedavg9` (for the Transformed Scheme II).

- If you don't want to use the Scheme I (where we sample device acccording to $p_k$ and simply average local parameters), please add `--noaverage`.

- If you want to mute the printed information, please use `--noprint`.

3. Once the trainning is started, logs that containning trainning statistics will be automatically created in ` result/$DATASET`. Each run has a unique log file name in this way `year-month-day-time_$ALGO_$NET_wn10_tn100_sd$SEED_lr$LR_ep$E_bs$B_a/w`, for example, 
    ```
   2019-11-24T12-05-13_fedavg4_logistic_wn10_tn100_sd0_lr0.1_ep5_bs64_a
   ```

4. During the trainning, you visualize the process by running either of the following

  ```
   tensorboard --logdir=result/$DATASET
   tensorboard --logdir=result/$DATASET/$LOG
   # For example
   tensorboard --logdir=result/mnist_all_data_0_equal_niid/
   tensorboard --logdir=result/mnist_all_data_0_equal_niid/2019-11-24T12-05-13_fedavg4_logistic_wn10_tn100_sd0_lr0.1_ep5_bs64_a
  ```

5. All the codes we used to draw figures are in `plot/`. You can find some choices of hyperparameters in both our paper and the scripts in `plot/`.



## Dependency

Pytorch = 1.0.0

numpy = 1.16.3

matplotlib = 3.0.0

tensorboardX





## data preparation
- To generate MNIST data for pretrain, please run "python data/mnist/generate_random_niid.py"
- To generate MNIST-M data for finetune, please download an mnist_m.tar.gz file from "https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg" to the path "./data/TARGET/mnist-m", and then unzip it:"cd data/TARGET/mnist-m", "tar -xf mnist_m.tar.gz"

## command
nohup python Transfer__MNIST_MNIST-M.py >> xx.log 2>&1 &

## result directory
Figure name: Source_Target_r_{federated round}_Fe_{federated epoch}_fe_{finetune epoch}
i.e., MNIST_MNIST-M_r_200_Fe_5_fe_100

## baseline
model_source_only: federatedly train on source domain
model_target_only: train on target domain
model_flag: randomly initialized the model and freeze it, finetune the last layer on target domain
model_ft: federatedly train on source domain, finetune the last layer on target domain.

## Aug 30 hyper-parameter
--num_epoch 10 --lr 0.01 --wd 0.0 --n_round 200 --finetune_epochs 100


