import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataset import *
from torchsummary import summary
from models import fastvit
import matplotlib.pyplot as plt
import numpy as np
# For reproducibility
torch.manual_seed(1)

if __name__ == '__main__':
    
    batch_size = 128
    
    force_=[]
    touch_ = []
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = fastvit.egotouch()
    
    force_criterion = nn.MSELoss()
    touch_criterion = nn.CrossEntropyLoss()
    
    evalsets = CustomImageDataset('dataset', 'eval')
    evalloader = DataLoader(evalsets,batch_size, shuffle=False)
    
    n = len(evalloader)
    
    with torch.no_grad():
        square_sum = 0
        num_instances = 0
        model.eval()
        for data in evalloader:
            inputs, r_theta, targets = data
            force_targets, touch_targets = torch.chunk(targets,2, dim = 1)

            force_outputs, touch_outputs = model(inputs.to(device), r_theta.to(device))
            force_.append(force_outputs)
            touch_.append(touch_outputs)
            
            square_sum += torch.sum((force_outputs- force_targets) ** 2).item() 
            square_sum += torch.sum((touch_outputs - touch_targets) ** 2).item()
            num_instances += len(targets)
            
          
            
    # plt.figure(figsize=(10, 8))
    
    # plt.subplot(121)
    # plt.plot()
    # plt.title("Training Loss")
    # plt.xlabel("epoch")
    # plt.show()
    # plt.savefig('loss_graph.jpg')
    print("RMSE of Eval sets : %.5f" % np.sqrt(square_sum / num_instances))