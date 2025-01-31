# Tygron AI Suite
This is a simple package for training [Mask R-CNN](https://pytorch.org/vision/main/models/mask_rcnn.html) networks using datasets exported from the [Tygron Platform](www.tygron.com).


# Contents
This repository contains a few simple classes and functions for training a simple Mask R-CNN for object detection on satellite images. Additionally it contains functions to export the model to [ONNX](https://onnx.ai/) with parameters and metadata such that it can be imported easily into the Tygron Platform.

The Configuration class stores the settings for:
* The train and test dataset location
* Mask R-CNN model parameters:
 * Channels
* Settings for exporting the PyTorch model to ONNX
* Metadata that is interpreted by the Tygron Platform, such as:
  * Attributes
  * Prefered pixel sizes
  * Description, producer and version
  * Legend entries

It also has a dataset class that simplifies training based on the data generated from projects based in the Tygron Platform.

# Installation
The easiest way to get started is downloading and using anaconda. A conda enviroment yml-file is provided in this repository to automatically setup a conda environment that contains all packages used in the provided jupyter notebook.

## Anaconda
[Download](https://www.anaconda.com/download/) Anaconda and install it to access conda and the anaconda-navigator.

For Linux and Mac users, it might be required to follow the extra steps described at (https://docs.anaconda.com/anaconda/install/)

## Import and setup environment automatically
Once installed, open the Anaconda Navigator app. Once open, make sure the application is updated to the latest version. 
Next, select the environments tab and in the bar below, select the option import an environment from a local file. Select the tygronai.yml file provided by this repository. Please note that importing this environment may take a while.

When the import is completed, you can optionally select this environment as your default. In the upper left corner of the Anaconda-navigator app, select File > Preferences. For the option "Default conda environment" select tygronai.

# Ubuntu shell script for starting anaconda-navigator
To activate the environment and start anaconda-navigator using a simple shell script file, create one, for example named '''anaconda.sh''', with the following commands:
```
conda init 
anaconda-navigator
```
Call the shell script using ```bash -i anaconda.sh```

In case conda is not found (writing conda --version in command terminal also fails), use the following lines instead:
```
# Replace <PATH_TO_CONDA> with the path to your conda install
source <PATH_TO_CONDA>/bin/activate
conda init 
anaconda-navigator
```
In case Anaconda was installed at /home/user/anaconda3, the <PATH_TO_CONDA> is /home/user/anaconda3/bin/conda

# Getting Started

* Export AI Datasets from Tygron Platform projects. Make sure you have a separated the folders containing train datasets and test datasets.
* Start Anaconda Navigator.
* Open the Jupyter Lab app (if necessary, install it first).
* Go to local directory of this repository.
* Open the example notebook named example_config.ipynb.
* Adjust the following path variables:
  * trainDirectory = "PATH TO TRAIN FILES"
  * testDirectory = "PATH TO TEST FILES"
* Press the "Restart kernel and execute all cells" button.

# Manual setup of a conda environment
In case you want to manually setup the conda enviroment, for example when you want to combine it with an existing environment, follow the instructions below.

## Create conda enviroment
```
conda create --name tygronai python=3.11
```
Once create, activate itActivate it
```
conda activate tygronai
````
## Install pytorch and torchvision with cuda:
In case you have a GPU that supports at least CUDA 12.1, you can run the conda install instructino below.
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
For alternative versions of pytorch, such as the cpu only version, please refer to the [official instructions](https://pytorch.org/get-started/locally/)

## Install onnx and onnxruntime
To import a trained neural network into the Tygron Platform, it needs to be exported from [PyTorch to ONNX](https://pytorch.org/docs/stable/onnx.html). The following packages need to be installed.
```
conda install onnx onnxruntime
```
## Install onnxscript
```
pip install onnxscript
```
