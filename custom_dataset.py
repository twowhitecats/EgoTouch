import torch
import pandas as pd
import os
import math
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image




class CustomImageDataset(Dataset):
    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = TorchStandardScaler()
        
        
        # image path list
        self.data_path = os.path.join(self.root_dir,self.mode, 'image')
        self.image_file_list = sorted([os.path.join(root,file) for root, _, files in os.walk(self.data_path) for file in files])
        
        self.img_T =  T.Compose([
            T.Resize((256,256)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5,),(0.5,))])
        
        # label csv file
        df_all = pd.DataFrame(columns=['force', 'touch'])
        csv_list = sorted([os.path.join(root,file) for root, _, files in os.walk(os.path.join(self.root_dir, mode,'label')) for file in files])
        print('list of label\'s csv file : ',csv_list)
        for csv in csv_list:
            df = pd.read_csv(csv, index_col=0)
            print(df.head(10))
            
            df_all = pd.concat([df_all, df],axis = 0, ignore_index=True)
            
        self.label_list = torch.tensor(df_all.iloc[ :,:2].values.astype(int), dtype = torch.float32)
        
        if self.transform:
            self.transform.fit(self.label_list)
            self.label_list = self.transform.transform(self.label_list)
        print('head of CustomImageDataset\'s Label',self.label_list[:10], '\n')
        
        
        #r, theta csv file
        df_angle = pd.DataFrame(columns=['r','theta'])
        angle_list = sorted([os.path.join(root,file) for root, _, files in os.walk(os.path.join(self.root_dir, mode,'angle')) for file in files])
        for csv in angle_list:
            df = pd.read_csv(csv, index_col =0)
            print(df.head(10))
            df_angle = pd.concat([df_angle,df],axis = 0, ignore_index=True)
        print('head of CusomAngleDataset\'s Angle', df_angle.head(10))
        self.angle_list = torch.tensor(df_angle.iloc[:,:2].values.astype(int), dtype = torch.float32)

        
        if self.transform:
            self.transform.fit(self.angle_list)
            self.angle_list = self.transform.transform(self.angle_list)
            
        print(len(self.image_file_list), ", ", len(self.label_list),", ", len(self.angle_list))
        assert len(self.image_file_list) == len(self.label_list) == len(self.angle_list)
        
    def __len__(self):
        return len(self.image_file_list)
    
    def __getitem__(self,idx):
        image = Image.open(self.image_file_list[idx])
        image_tensor = self.img_T(image)
        
        angle = torch.FloatTensor(self.angle_list[idx])
        label = self.label_list[idx]
        return image_tensor, angle, label
        
class CustomAngleDataset(Dataset):
    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        
        self.transform = TorchStandardScaler()
        
        self.labeldir = os.path.join(self.root_dir, mode, 'label')
        self.angledir = os.path.join(self.root_dir, mode, 'angle')
        
        df_angle = pd.DataFrame(columns=['r','theta'])
        angle_list = sorted([os.path.join(root,file) for root, _, files in os.walk(os.path.join(self.root_dir, mode,'sensor')) for file in files])
        print("list of angle: ", angle_list)
        for csv in angle_list:
            df = pd.read_csv(csv)
            df_angle = pd.concat([df_angle,df], ignore_index=True)
        print('head of CusomAngleDataset\'s Angle', df_angle.head(10))
        self.angle_list = torch.tensor(df_angle.iloc[:,:2].values())
        
        if self.transform:
            self.transform.fit(self.angle_list)
            self.transform.transform(self.angle_list)
            
        df_label = pd.DataFrame(columns= ['label'])
        label_list = sorted([os.path.join(root,file) for root, _, files in os.walk(os.path.join(self.root_dir, mode,'label')) for file in files])
        print("list of label: " ,label_list)
        for csv in label_list:
            df = pd.read_csv(csv)
            df_label = pd.concat([df_label, df], ignore_index=True)
        print('head of CustomImageDataset\'s Label',df_label.head(10))
        
        self.label_list =torch.tensor(df_label.iloc[:,1].values())
        
        assert self.angle_list.shape[0] == len(self.label_list)
            
    def __len__(self):
        return len(self.angle_list.shape[0])
    
    def __getitem__(self,idx):
        angle = torch.FloatTensor(self.angle_list[idx])
        label = torch.FloatTensor(self.label_list[idx])
        return angle, label
            
class TorchStandardScaler:
  def fit(self, x):
    self.mean = x.mean(0, keepdim=True)
    self.std = x.std(0, unbiased=False, keepdim=True)
  def transform(self, x):
    x -= self.mean
    x /= (self.std + 1e-7)
    return x


# foo = TorchStandardScaler()
# foo.fit(data)
# print(f"mean {foo.mean}, std {foo.std}")
# foo.transform(data)
#https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/7