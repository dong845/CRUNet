import numpy as np
from torch.utils.data import Dataset
import os
import h5py
from numpy.fft import fft, fft2, ifftshift, fftshift, ifft2
import torch

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


class CineDataset_MC_Philips_New(Dataset):
    def __init__(self, files, transform=None):
        super().__init__()
        self.files = files
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
        img = self.ifft2c(kspace)
        img = img/np.max(np.abs(img))
        kspace = self.fft2c(img)
        return kspace
         
    def __getitem__(self, idx):
        file_path = self.files[idx]
        name = file_path.split('/')[-1].split('.')[0]
        
        with h5py.File(file_path, 'r') as h5file:
            print("Keys: %s" % h5file.keys())
            print("Keys: %s" % h5file['AllData'].keys())
        
            bookkeeping = h5file['AllData']['bookkeeping'][:]
            csm = h5file['AllData']['csm_r'][:] + 1j*h5file['AllData']['csm_r'][:]
            kspace = h5file['AllData']['indata_raw_r'][:] + 1j*h5file['AllData']['indata_raw_i'][:]
            temp = h5file['AllData']['indata_raw_r']
            inv_sqrt_reg = np.sqrt(-1 * h5file['AllData']['mininvreg'][:] )
        
        
            nrCardPhases = int(bookkeeping[:,0].max()+1)
            nrLocations = int(bookkeeping[:,1].max()+1)
            
            kspace_ordered = np.zeros_like(kspace, shape = (nrCardPhases, nrLocations, kspace.shape[1], kspace.shape[2], kspace.shape[3]))
            csm_ordered = np.zeros_like(kspace, shape = (nrCardPhases, nrLocations, csm.shape[1], csm.shape[2],csm.shape[3]))
            inv_sqrt_reg_ordered = np.zeros_like(inv_sqrt_reg, shape = (nrCardPhases, nrLocations, inv_sqrt_reg.shape[1], inv_sqrt_reg.shape[2]))
            for ind in range(bookkeeping.shape[0]):
                kspace_ordered[int(bookkeeping[ind,0]),int(bookkeeping[ind,1]),...] = kspace[ind,...]
                csm_ordered[int(bookkeeping[ind,0]),int(bookkeeping[ind,1]),...] = csm[ind,...]
                inv_sqrt_reg_ordered[int(bookkeeping[ind,0]),int(bookkeeping[ind,1]),...] = inv_sqrt_reg[ind,...]   
        
        kspace_ordered = torch.fft.fftshift(torch.tensor(kspace_ordered), dim=(-2,-1))
        mask = (np.abs(kspace_ordered.numpy()) > 0).astype(np.uint8)[:, 0]
        und_kspace = self.norm(kspace_ordered)[:,0]  # delete the slice dimension
        sense_map = csm_ordered[:,0]
        return und_kspace, mask, sense_map, name
