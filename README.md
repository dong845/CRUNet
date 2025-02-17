# CRUNet

Code Implementation of CRUNet-MR method.

## File Description

- models/CRUNet_MR.py: Network structure of CRUNet-MR and an optimized version
- cine_dataset.py: Build dataset class (Fot the uploaded data, it is already processed before saving into the h5 files, so it is mainly a loading process inside the class.)
- requirements.txt: Some main python packages to be installed
- train_infer.py: Including training and testing.

## 🔨 Usage

For the inference of testing data: 
    1. set path for the variable **infer_weight_path** and **test_path** of args 
    2. **axis** and **mode** can be changed to the corresponding axis view and acceleration factor
    3. create folder for value files and set it to **save_val_path** of args
    4. then run python command
```
python train_infer.py
```
