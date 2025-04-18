# CRUNet

Code Implementation of CRUNet-MR method.

## File Description

- models/CRUNet_MR.py: Network structure of CRUNet-MR and an optimized version
- cine_dataset.py: Build dataset class 
   * For the uploaded CMRxRecon2023 data, it is already processed before saving into the h5 files, so it is mainly a loading process inside the class. Use "CineDataset_MC" class. 
   * For the testing Philips data, the input format should also be h5 file, with the keys ("FullSample", "mask", "sense_map", "UnderSample") inside it. They all have the shape of (frame_num, slice_num, coil_num, height, width) and slice_num=1. Use "CineDataset_MC_Philips" class.
- requirements.txt: Some main python packages to be installed
- train_infer.py: Including training and testing.

## 🔨 Usage

For the inference of testing data: 
- set path for the variable **infer_weight_path** (choose the weight file with "latest") and **test_path** of args 
- **axis** and **mode** can be changed to the corresponding axis view and acceleration factor
- create folder for value files and set it to **save_val_path** of args
- then run python command
```
python train_infer.py
```
