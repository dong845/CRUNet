import numpy as np
from torch.utils.data import Dataset
import os
import h5py
from numpy.fft import fft, fft2, ifftshift, fftshift, ifft2

class CineDataset_MC(Dataset):
    def __init__(self, files, folder_path, mode, transform=None):
        super().__init__()
        self.files = files
        self.folder_path = folder_path
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        file_path = os.path.join(self.folder_path, file)
        name = file.split('.')[0]
        with h5py.File(file_path, 'r') as f:
            full_kspace = np.array(f["FullSample"])
            mask = np.array(f[f"{self.mode}_mask"])
            sense_map = np.array(f["sense_map"])
            und_kspace = np.array(f[f"{self.mode}"])
        if self.transform is not None:
            return self.transform(full_kspace), self.transform(und_kspace), mask, sense_map, name
        return full_kspace, und_kspace, mask, sense_map, name

class CineDataset_MC_Philips(Dataset):
    def __init__(self, files, folder_path, mode, transform=None):
        super().__init__()
        self.files = files
        self.folder_path = folder_path
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def ifft2c(self, kspace):
        axes = (-2, -1)
        return fftshift(fft2(ifftshift(kspace, axes=axes), axes=axes, norm='ortho'), axes=axes)

    def fft2c(self, img):
        axes = (-2, -1)
        return fftshift(ifft2(ifftshift(img, axes=axes), axes=axes, norm='ortho'), axes=axes)
    
    def norm(self, kspace):
        img = self.ifft2c(kspace[:,0])
        img = img/np.max(np.abs(img))
        kspace = self.fft2c(img)
        return kspace
         
    def __getitem__(self, idx):
        file = self.files[idx]
        file_path = os.path.join(self.folder_path, file)
        name = file.split('.')[0]
        with h5py.File(file_path, 'r') as f:
            full_kspace = np.array(f["FullSample"])
            mask = np.array(f["mask"]) # t slice cn h w
            sense_map = np.array(f["sense_map"])
            und_kspace = np.array(f[f"UnderSample"])
        full_kspace = self.norm(full_kspace)
        und_kspace = self.norm(und_kspace)
        mask = mask[:,0]
        sense_map = sense_map[:,0]
        if self.transform is not None:
            return self.transform(full_kspace), self.transform(und_kspace), mask, sense_map, name
        return full_kspace, und_kspace, mask, sense_map, name
