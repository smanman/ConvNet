# An Efficient and Lightweight Adaptive Network for 3D Medical Image Segmentation

---
## Installation
#### 1. System requirements
The code is tested with PyTorch 1.11.0 and CUDA 11.3. After cloning the repository, follow the below steps for installation.
#### 2. Installation guide
We recommend installation of the required packages using the conda package manager, available through the Anaconda Python distribution. Anaconda is available free of charge for non-commercial use through [Anaconda Inc](https://www.anaconda.com/products/individual). After installing Anaconda and cloning this repository, For use as integrative framework：
```
conda create --name ConvNet++ python=3.8
conda activate ConvNet++
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

#### 3. Functions of scripts and folders
- **For evaluation:**
  - ``unetr_pp/inference_fdg_pet_ct.py``
  
  - ``unetr_pp/inference_synapse.py``
  
  - ``unetr_pp/inference_tumor.py``
  - 
  - ``unetr_pp/inference_lung.py``
  - 
  - ``unetr_pp/inference_btcv.py``
  
- **For inference:**
  - ``unetr_pp/inference/predict_simple.py``
  
- **Network architecture:**
  - ``unetr_pp/network_architecture/fdg_pet_ct/``
  
  - ``unetr_pp/network_architecture/synapse/``
  
  - ``unetr_pp/network_architecture/tumor/``
  
  - ``unetr_pp/network_architecture/lung/``
  
  - ``unetr_pp/network_architecture/btcv/``

- **For training:**
  - ``unetr_pp/run/run_training.py``
  
- **Trainer for dataset:**
  - ``unetr_pp/training/network_training/unetr_pp_trainer_pet.py``
  
  - ``unetr_pp/training/network_training/unetr_pp_trainer_synapse.py``
  
  - ``unetr_pp/training/network_training/unetr_pp_trainer_tumor.py``
  
  - ``unetr_pp/training/network_training/unetr_pp_trainer_lung.py``
  
  - ``unetr_pp/training/network_training/unetr_pp_trainer_btcv.py``
---

## Training
#### 1. Dataset download
Datasets can be acquired via following links:

**Dataset I**
[FDG_PET/CT](https://autopet.grand-challenge.org/)

**Dataset II**
[The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

**Dataset III**
[Brain_tumor](http://medicaldecathlon.com/)

**Dataset IV**
[Lung](https://drive.google.com/file/d/1KdEhz7hWjIQvmz5dCFwxlttpUFQJmCnz)


#### 2. Setting up the datasets
After you have downloaded the datasets, you can follow the settings in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:

```
./DATASET/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
          ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task03_tumor/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task04_BTCV/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task06_Lung/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task07_FDG_PET/CT/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
      ├── unetr_pp_cropped_data/
  ├── nnFormer_trained_models/
  ├── nnFormer_preprocessed/
```

#### 3. Training and Testing
- **For training:**
  - ``python run_training.py``
  
- **Test:**
  - ``python predict_simple.py``

  - ``python inferenct_dataset.py``

## 4. Contact
Should you have any question, please create an issue on this repository or contact at 2436381007@qq.com.


# ConvNet-
# ConvNet-
# ConvNet
