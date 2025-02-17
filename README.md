# CRUNet

Code Implementation of CRUNet-MR method.

## File Description

- models/CRUNet_MR.py: network structure of CRUNet-MR and an optimized version
- cine_dataset.py: Build dataset class (Fot the uploaded data, it is already processed before saving into the h5 files, so it is mainly a loading process inside the class.)
- requirements.txt: some main python packages to be installed
- train_infer.py: a file contains training and testing

## 🔨 Usage

For the inference of testing data, set path for the variable **infer_weight_path** and **test_path** of argparse. **axis** and **mode** can be changed to the corresponding axis view and acceleration factor, then

```
python train_infer.py
```
