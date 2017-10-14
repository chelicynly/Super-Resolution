import torch.utils.data as data
import torch
import h5py
import os.path
class DatasetFromHdf5(data.Dataset):
    def __init__(self, root_dir = './data', scale = 2,train = True, filename = None):
        super(DatasetFromHdf5, self).__init__()
        self.train = train
        if filename == None:
            filename = 'train_s'+str(scale)+'.h5'
            if self.train:
                filename = 'train_s'+str(scale)+'.h5'
            else:
                filename = 'test_s'+str(scale)+'.h5'
        else:
            filename = filename
        file_path = os.path.join(root_dir, filename)
        hf = h5py.File(file_path)
        self.data = torch.from_numpy(hf.get('data').value)
        self.target = torch.from_numpy(hf.get('label').value)

    def __getitem__(self, index):  
                  
        return self.data[index,:,:,:].float(), self.target[index,:,:,:].float()
        
    def __len__(self):
        return self.data.shape[0]
