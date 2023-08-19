# Lightning Emulation Code
The Lightning emulation code evaluates the impact of errors in the photonic domain on large DNNs, which performs inference with 8-bit photonic, 8-bit digital, and 32-bit digital computation schemes. The code in this folder reproduces results reported in Lightning paper Section 7.

## Folder structure
|  Source Files               |Description                                                                                                                    |
|  -----                      |  -----                                                                                                                          |
|  `models_weight/` |  Before running experiments, put the model weight files in this folder based on the instructions below.|
|  `results/` |  The final emulation results after running experiments will be stored here.                         |
|  `emulate_alexnet.py` |  The python script to reproduce AlexNet emulation results.                         |
|  `emulate_vgg11.py` |  The python script to reproduce VGG11 emulation results.                         |
|  `emulate_vgg16.py` |  The python script to reproduce VGG16 emulation results.                         |
|  `emulate_vgg19.py` |  The python script to reproduce VGG19 emulation results.                         |
|  `README.md` |  This file, describing the required libraries and instructions to reproduce the experiments.                |

## Required python version and libraries
- Python version: Python 3.10.8
- Libraries:
    1. torch
    2. numpy
    3. torchvision
    4. enum
    5. math

## Steps to reproduce the emulation results

### 1. Prepare ImageNet dataset
Download [ImageNet dataset](https://www.image-net.org/download.php), follow the instructions to extract the valiadation directory. And then replace the ```val_dir``` parameter in each python script to be your own ImageNet validation path.

### 2. Prepare model weights
Download model weights from [this link](https://drive.google.com/drive/folders/1tmVsB3pa4_GFe2ezk7MxVSjKP74jKNxH?usp=drive_link). Download ```alexnet.pth```, ```vgg11.pth```, ```vgg16.pth```,  ```vgg19.pth```, and put the four model weight files in the ```model_weight/``` folder.

### 3. Reproduce AlexNet emulation results
Run ```python emulate_alexnet.py``` <br>
After the experiments, the results will be stored in ```results/``` folder:
- ```results/alexnet_digital_32bit.txt```: Inference accuracy results of AlexNet on Digital-32bit accelerators
- ```results/alexnet_digital_8bit.txt```: Inference accuracy results of AlexNet on Digital-8bit accelerators
- ```results/alexnet_lightning.txt```: Inference accuracy results of AlexNet on Lightning

### 4. Reproduce VGG11 emulation results
Run ```python emulate_vgg11.py``` <br>
After the experiments, the results will be stored in ```results/``` folder:
- ```results/vgg11_digital_32bit.txt```: Inference accuracy results of VGG11 on Digital-32bit accelerators
- ```results/vgg11_digital_8bit.txt```: Inference accuracy results of VGG11 on Digital-8bit accelerators
- ```results/vgg11_lightning.txt```: Inference accuracy results of VGG11 on Lightning

### 5. Reproduce VGG16 emulation results
Run ```python emulate_vgg16.py``` <br>
After the experiments, the results will be stored in ```results/``` folder:
- ```results/vgg16_digital_32bit.txt```: Inference accuracy results of VGG16 on Digital-32bit accelerators
- ```results/vgg16_digital_8bit.txt```: Inference accuracy results of VGG16 on Digital-8bit accelerators
- ```results/vgg16_lightning.txt```: Inference accuracy results of VGG16 on Lightning

### 6. Reproduce VGG19 emulation results
Run ```python emulate_vgg19.py``` <br>
After the experiments, the results will be stored in ```results/``` folder:
- ```results/vgg19_digital_32bit.txt```: Inference accuracy results of VGG19 on Digital-32bit accelerators
- ```results/vgg19_digital_8bit.txt```: Inference accuracy results of VGG19 on Digital-8bit accelerators
- ```results/vgg19_lightning.txt```: Inference accuracy results of VGG19 on Lightning