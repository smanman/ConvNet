# An Efficient and Lightweight Adaptive Network for 3D
Medical Image Segmentation

## Overview

 In this study, we revaluate and improve the architecture of Convolutional Neural Networks (CNNs) by constructing a lightweight three-dimensional convolutional network (ConvNet++). We revisit volumetric convolution through the concept of large-kernel deep convolution modules and implement independent linear depth convolution scaling for each channel feature. This improvement not only contributes to the architecture of deep learning models in the field of artificial intelligence but also provides a more efficient solution for medical image segmentation.We design a feature fusion method to better utilize multi-scale spatial information, avoiding simple addition or concatenation, thereby alleviating issues related to excessive model parameters and ineffective feature fusion. Additionally, we introduce a Lightweight Hybrid Attention (LHA) module that employs a self-attention mechanism with a shared key-query scheme to efficiently encode spatial and channel information. The LHA module facilitates effective communication between spatial and channel branches, providing complementary features while reducing the overall number of parameters. This design represents an innovation in adaptive modules within the field of artificial intelligence, enabling better capture of subtle features in medical images. We conduct extensive evaluations on four challenging benchmark datasets, and the results indicate that our contributions are highly effective in terms of parameter efficiency and accuracy.


## System Requirements

- **Python**: 3.8 or later
- **PyTorch**: 1.11.0 (tested configuration)
- **CUDA**: 11.3 (for GPU acceleration)
- **Operating System**: Platform-independent (tested on Linux and Windows)

## Installation Guide

### Environment Setup

We recommend using the Anaconda package management system for installation. Anaconda is available for non-commercial use through [Anaconda Inc](https://www.anaconda.com/products/individual).

```bash
# Create and activate a dedicated conda environment
conda create --name ConvNet++ python=3.8
conda activate ConvNet++

# Install PyTorch with CUDA support
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install additional dependencies
pip install -r requirements.txt
```

## Repository Structure

### Evaluation Modules
- `unetr_pp/inference_fdg_pet_ct.py` - Evaluation pipeline for FDG PET/CT datasets
- `unetr_pp/inference_synapse.py` - Evaluation pipeline for Synapse multi-organ datasets
- `unetr_pp/inference_tumor.py` - Evaluation pipeline for brain tumor datasets
- `unetr_pp/inference_lung.py` - Evaluation pipeline for lung datasets
- `unetr_pp/inference_btcv.py` - Evaluation pipeline for BTCV datasets

### Inference Module
- `unetr_pp/inference/predict_simple.py` - Interface for single-case inference and deployment

### Network Architecture Implementations
- `unetr_pp/network_architecture/fdg_pet_ct/` - Specialized architecture for PET/CT imaging
- `unetr_pp/network_architecture/synapse/` - Specialized architecture for multi-organ segmentation
- `unetr_pp/network_architecture/tumor/` - Specialized architecture for brain tumor segmentation
- `unetr_pp/network_architecture/lung/` - Specialized architecture for lung segmentation
- `unetr_pp/network_architecture/btcv/` - Specialized architecture for BTCV segmentation

### Training Framework
- `unetr_pp/run/run_training.py` - Primary training execution script

### Dataset-Specific Training Implementations
- `unetr_pp/training/network_training/unetr_pp_trainer_pet.py` - Training pipeline for PET/CT datasets
- `unetr_pp/training/network_training/unetr_pp_trainer_synapse.py` - Training pipeline for Synapse datasets
- `unetr_pp/training/network_training/unetr_pp_trainer_tumor.py` - Training pipeline for brain tumor datasets
- `unetr_pp/training/network_training/unetr_pp_trainer_lung.py` - Training pipeline for lung datasets
- `unetr_pp/training/network_training/unetr_pp_trainer_btcv.py` - Training pipeline for BTCV datasets

---

## Implementation Guide

### Dataset Acquisition

The framework has been evaluated on the following publicly available datasets:

| Dataset | Imaging Modality | Link |
|---------|-----------------|------|
| Dataset I | FDG PET/CT | [AutoPET Challenge](https://autopet.grand-challenge.org/) |
| Dataset II | Multi-organ CT | [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) |
| Dataset III | Brain MRI | [Medical Decathlon](http://medicaldecathlon.com/) |
| Dataset IV | Lung CT | [Download Link](https://drive.google.com/file/d/1KdEhz7hWjIQvmz5dCFwxlttpUFQJmCnz) |

### Dataset Configuration

After acquisition, configure the datasets following the preprocessing guidelines established in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md). The directory structure should be organized as follows:

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

### Execution Instructions

#### Training
To initiate the model training process:
```bash
python run_training.py
```

#### Inference & Evaluation
For single case inference:
```bash
python predict_simple.py
```

For dataset-level evaluation:
```bash
python inference_dataset.py
```

## Contributing

We welcome contributions to enhance ConvNet++ functionality. Please follow standard GitHub practices for pull requests and issue reporting.

## Contact Information

For technical inquiries or collaboration opportunities, please:
- Create an issue on this repository
- Contact the development team at 2436381007@qq.com

