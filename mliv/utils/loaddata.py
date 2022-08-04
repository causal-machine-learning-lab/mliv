import pandas as pd
import copy
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except:
    print('No module named torch. Please pip install torch')

def get_var_df(df,var):
    var_cols = [c for c in df.columns if c.startswith(var)]
    return df[var_cols].to_numpy()

def cat(data_list, axis=1):
    try:
        output=torch.cat(data_list,axis)
    except:
        output=np.concatenate(data_list,axis)

    return output

def split(data, split_ratio=0.5):
    data1 = copy.deepcopy(data)
    data2 = copy.deepcopy(data)

    split_num = int(data.length * split_ratio)
    data1.split(0, split_num)
    data2.split(split_num, data.length)

    return data1, data2

class CausalDataset(object):
    def __init__(self, path):
        self.path  = path 
        self.train = getDataset(pd.read_csv(path + 'train.csv'))
        self.valid = getDataset(pd.read_csv(path + 'valid.csv'))
        self.test  = getDataset(pd.read_csv(path + 'test.csv'))

    def split(self, split_ratio=0.5, data=None):
        if data is None:
            data = self.train

        data1, data2 = split(data, split_ratio)
        self.data1 = data1
        self.data2 = data2

    def get_train(self):
        return self.train

    def get_valid(self):
        return self.valid

    def get_test(self):
        return self.test

    def get_data(self):
        return self.train,self.valid,self.test

    def tensor(self):
        self.train.tensor()
        self.valid.tensor()
        self.test.tensor()

    def double(self):
        self.train.double()
        self.valid.double()
        self.test.double()

    def float(self):
        self.train.float()
        self.valid.float()
        self.test.float()

    def detach(self):
        self.train.detach()
        self.valid.detach()
        self.test.detach()

    def to(self, device='cpu'):
        self.train.to(device)
        self.valid.to(device)
        self.test.to(device)

    def cpu(self):
        self.train.cpu()
        self.valid.cpu()
        self.test.cpu()

    def numpy(self):
        self.train.numpy()
        self.valid.numpy()
        self.test.numpy()

class TorchDataset(Dataset):
    def __init__(self, data, device='cpu', type='tensor'):
        if type == 'tensor':
            data.tensor()
        else:
            data.double()
        data.to(device)
        
        self.data = data
    
    def __getitem__(self, idx):
        var_dict = {}
        for var in self.data.Vars:
            exec(f'var_dict[\'{var}\']=self.{var}[idx]')
        
        return var_dict

    def __len__(self):
        return self.data.length

class getDataset(Dataset):
    def __init__(self, df):
        self.length = len(df)
        self.Vars = list(set([col[0] for col in df.columns]))

        for var in self.Vars:
            exec(f'self.{var}=get_var_df(df, \'{var}\')')

        if not hasattr(self, 'i'):
            self.i = self.z
            self.Vars.append('i')

    def split(self, start, end):
        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}[start:end]')
            except:
                pass

        self.length = end - start

    def cpu(self):
        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}.cpu()')
            except:
                break
    
    def cuda(self,n=0):
        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}.cuda({n})')
            except:
                break

    def to(self,device='cpu'):
        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}.to(\'{device}\')')
            except:
                break
    
    def tensor(self):
        for var in self.Vars:
            try:
                exec(f'self.{var} = torch.Tensor(self.{var})')
            except:
                break

    def float(self):
        for var in self.Vars:
            try:
                exec(f'self.{var} = torch.Tensor(self.{var}).float()')
            except:
                break    
            
    def double(self):
        for var in self.Vars:
            try:
                exec(f'self.{var} = torch.Tensor(self.{var}).double()')
            except:
                break

    def detach(self):
        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}.detach()')
            except:
                break
            
    def numpy(self):
        try:
            self.detach()
        except:
            pass

        try:
            self.cpu()
        except:
            pass

        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}.numpy()')
            except:
                break

    def pandas(self, path=None):
        var_list = []
        var_dims = []
        var_name = []
        for var in self.Vars:
            exec(f'var_list.append(self.{var})')
            exec(f'var_dims.append(self.{var}.shape[1])')
        for i in range(len(self.Vars)):
            for d in range(var_dims[i]):
                var_name.append(self.Vars[i]+str(d))
        df = pd.DataFrame(np.concatenate(var_list, axis=1),columns=var_name)

        if path is not None:
            df.to_csv(path, index=False)
        return df

    def __getitem__(self, idx):
        var_dict = {}
        for var in self.Vars:
            exec(f'var_dict[\'{var}\']=self.{var}[idx]')
        
        return var_dict

    def __len__(self):
        return self.length