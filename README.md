# Tygron AI Suite
This is a simple package for training [Mask R-CNN](https://pytorch.org/vision/main/models/mask_rcnn.html) networks using datasets exported from the [Tygron Platform](www.tygron.com).


# Contents
This repository contains a set of classes and functions for training a simple Mask R-CNN for object detection on satellite images. Additionally it contains functions to export the model to [ONNX](https://onnx.ai/) with parameters and metadata such that it can be imported easily into the Tygron Platform.

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
To get started, first download this tygron-ai-suite repository on your computer. Either use 
```
git clone
```
or download the repository as a zip and unzin it on your pc. 

To setup a python development enviroment for the Tygron AI Suite, we advice using Conda to setup an environment and manage python dependencies. Conda can be installed using [Conda-Forge](https://conda-forge.org), which is [open-source](https://github.com/conda-forge/miniforge).

A conda enviroment yml-file is provided in this repository to automatically setup a conda environment that contains all packages used in the provided jupyter notebook.

## Conda forge
Miniforge is the preferred conda-forge installer and includes conda, mamba, and their dependencies. You can download Miniforge from  [Conda-forge](https://conda-forge.org/download/). Once installed, you can open the Miniforge prompt to create a Conda environment.

Once installed, you can start using conda. Either open the terminal or start the Miniforge Prompt application.

## Conda enviroment
By default, conda opens with a base environment. To setup a new conda environment for the Tygron AI suite, change the working directory of the mini forge prompt to the download location of this repository. The tygronai.yml should be present in this directory.

Next, automatically setup a tygronai environment by typing:
```
conda env create -n tygronai -f tygronai.yml
```
This creates an enviroment named tygronai using the dependencies listed in the tygronai.yml file. Once the enviroment is set up, you can activate it by typing:
```
conda activate tygronai
```
And you can open a notebook by typing
```
jupyter notebook
```

# Getting Started

* Export AI Datasets from Tygron Platform projects. Make sure you have a separated the folders containing train datasets and test datasets.
* Start Mini-forge, activate the tygronai enviroment and navigate to the local tygron ai suite repository. See instructions below.
* Open the Jupyter Notebook app, by typing jupyter notebook.
* Open the example notebook named example_config.ipynb.
* Adjust the following path variables:
  * trainDirectory = "PATH TO TRAIN FILES"
  * testDirectory = "PATH TO TEST FILES"
* Press the "Restart kernel and execute all cells" button.

# Ubuntu (linux) shell script for starting anaconda-navigator
To activate the tygronai environment and start jupyter notebook app in your browser on linux using a simple shell script file, create one, for example named '''miniforge.sh''', with the following commands:
```
conda init 
conda activate tygronai
jupyter notebook
```
Call the shell script using ```bash -i miniforge.sh```

In case conda is not found (writing conda --version in command terminal also fails), use the following lines instead:
```
# Replace <PATH_TO_CONDA> with the path to your conda install
source <PATH_TO_CONDA>/bin/activate
conda init 
anaconda-navigator
```
In case Anaconda was installed at /home/user/anaconda3, the <PATH_TO_CONDA> is /home/user/anaconda3/bin/conda


