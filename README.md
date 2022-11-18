

# FedFly: Towards Migration in Edge-based Distributed Federated Learning

### About the research

Due to mobility, a device participating in Federated Learning (FL) may disconnect from one edge server and will need to connect to another edge server during FL training. This becomes more challenging when a Deep Neural Network (DNN) is partitioned between device and edge server referred to as edge-based FL. Moving a device without migrating the accompanying training data from a source edge server to the destination edge server will result in training for the device having to start all over again on the destination server. This will in turn affect the performance of edge-based FL and result in large training times. FedFly addresses the  mobility challenge of devices in edge-based distributed FL. This research designs, develops and implements the technique for migrating DNN in the context of edge-based distributed FL.

FedFly is implemented and evaluated in a hierarchical cloud-edge-device architecture on a lab-based testbed to validate the migration technique of edge-based FL. The testbed that includes four IoT devices, two edge servers, and one central server (cloud-like) running the VGG-5 DNN model. The empirical findings uphold and validates our claims in terms of training time and accuracy using balanced and imbalanced datasets when compared to state-of-the-art approaches, such as SplitFed. FedFly has a negligible overhead of up to 2 seconds but saves a significant amount of training time while maintaining accuracy.

<p align="center">
  <img src="FedFly System.png" alt="FedFly System width="200"/>
</p>

More information on the steps in relation to distributed FL and the mobility of devices within the FedFly system are presented in the research article entitled, [FedFly: Towards Migration in Edge-based Distributed Federated Learning](https://arxiv.org/pdf/2111.01516.pdf), IEEE Communications Magazine, 2022.
                                                             
### Code Structure

The repository contains the source code of FedFly. The overall architecture is divided as follows: 

1) Central server (Central server, such as a cloud location, for running the FedAverage algorithm)
2) Edge servers (separated as Source and Destination for migration)
3) Devices

The repository also arranges the code according to the above described architecture.

The results are saved as pickle files in the `results` folder on the Central Server. 


Currently, CIFAR10 dataset and Convolutional Neural Network (CNN) models are supported. The code can be extended to support other datasets and models.

### Setting up the environment

The code is tested on Python 3 with Pytorch version 1.4 and torchvision 0.5. 

In order to test the code, install Pytorch and torchvision on each IoT device (for example, Raspberry Pis as used in this work). One can install from pre-built PyTorch and torchvision pip wheel. Download respective pip wheel as follows:
- Pyotrch: https://github.com/FedML-AI/FedML-IoT/tree/master/pytorch-pkg-on-rpi

Or visit https://github.com/Rehmatkhan/InstallPytrochScript and follow the simple steps:
```
# install and configure pytorch and torchvision on Raspberry devices
#move to sudo
sudo -i
#update
apt update
apt install git
git clone https://github.com/Rehmatkhan/InstallPytrochScript.git
mv InstallPytrochScript/install_python_pytorch.sh .
chmod +x install_python_pytorch.sh
rm -rf InstallPytrochScript
./install_python_pytorch.sh
```

All configuration options are given in `config.py` at the central server, which contains the architecture, model, and FL training hyperparameters.
Therefore, modify the respective hostname and ip address in `config.py`. CLIENTS_CONFIG and CLIENTS_LIST in `config.py` are used for indexing and sorting.
Note that `config.py` file must be changed at the source edge server, destination edge server and at each device.  



```
# Network configration
SERVER_ADDR= '192.168.10.193'
SERVER_PORT = 51000
UNIT_MODEL_SERVER = '192.168.10.102'
UNIT_PORT = 51004

EDGE_SERVERS = {'Sierra.local': '192.168.10.193', 'Rehmats-MacBook-Pro.local':'192.168.10.154'}


K = 4 # Number of devices

# Unique clients order
HOST2IP = {'raspberrypi3-1':'192.168.10.93', 'raspberrypi3-2':'192.168.10.31', 'raspberrypi4-1': '192.168.10.169', 'raspberrypi4-2': '192.168.10.116'}
CLIENTS_CONFIG= {'192.168.10.93':0, '192.168.10.31':1, '192.168.10.169':2, '192.168.10.116':3 }
CLIENTS_LIST= ['192.168.10.93', '192.168.10.31', '192.168.10.169', '192.168.10.116'] 

```
Finally, download the CIFAR10 datasets manually and put them into the `datasets/CIFAR10` folder (python version). 
- CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html


To test the code:

#### Launch FedFly central server

```
python FedFly_serverrun.py --offload True #FedFly training
```
#### Launch FedFly source edge server

```
python FedFly_serverrun.py --offload True #FedFly training
```
#### Launch FedFly destination edge server
```
python FedFly_serverrun.py --offload True #FedFly training
```
#### Launch FedFly devices
```
python FedFly_clientrun.py --offload True #FedFly training
```

### Citation

Please cite the paper as follows: 

Rehmat Ullah, Di Wu, Paul Harvey, Peter Kilpatrick, Ivor Spence and Blesson Varghese, "FedFly: Toward Migration in Edge-Based Distributed Federated Learning," in IEEE Communications Magazine, vol. 60, no. 11, pp. 42-48, November 2022, doi: 10.1109/MCOM.003.2100964.

```
@ARTICLE{9846940,
  author={Ullah, Rehmat and Wu, Di and Harvey, Paul and Kilpatrick, Peter and Spence, Ivor and Varghese, Blesson},
  journal={IEEE Communications Magazine}, 
  title={FedFly: Toward Migration in Edge-Based Distributed Federated Learning}, 
  year={2022},
  volume={60},
  number={11},
  pages={42-48},
  doi={10.1109/MCOM.003.2100964}}
```
