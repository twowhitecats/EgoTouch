import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataset import *
from torchsummary import summary
from models import fastvit
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.classification import BinarySpecificity, ROC
# For reproducibility
torch.manual_seed(1)

if __name__ == '__main__':
    
    batch_size = 128
    
    force_=[]
    touch_ = []
    touch_target_ = []
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = fastvit.egotouch()
    model.eval()
    
    model.cuda()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    force_criterion = nn.MSELoss()
    touch_criterion = nn.CrossEntropyLoss()
    
    evalsets = CustomImageDataset('dataset', 'eval')
    evalloader = DataLoader(evalsets,batch_size, shuffle=False)
    
    n = len(evalloader)
    
    with torch.no_grad():
        square_sum = 0
        num_instances = 0
        start_event.record()
        
        for data in evalloader:
            inputs, r_theta, targets = data
            force_targets, touch_targets = torch.chunk(targets,2, dim = 1)

            force_outputs, touch_outputs = model(inputs.to(device), r_theta.to(device))
            force_+= force_outputs.tolist()
            touch_ += touch_outputs.tolist()
            touch_target_ += touch_targets.tolist()
            
            square_sum += torch.sum((force_outputs- force_targets.to(device)) ** 2).item() 
            square_sum += torch.sum((touch_outputs - touch_targets.to(device)) ** 2).item()
            num_instances += len(targets)
        end_event.record()
    torch.cuda.synchronize()
    time_taken = start_event.elapsed_time(end_event)
            
            
    print('Time Took for ', n,' datasets' ,' :' ,time_taken)
    
    specificity = BinarySpecificity() 
    specificity(torch.tensor(touch_),torch.tensor(touch_target_))
    print("touch false positive rate : ", specificity.item())
            
    # plt.figure(figsize=(10, 8))
    
    # plt.subplot(121)
    # plt.plot()
    # plt.title("Training Loss")
    # plt.xlabel("epoch")
    # plt.show()
    # plt.savefig('loss_graph.jpg')
    print("RMSE of Eval sets : %.5f" % np.sqrt(square_sum / num_instances))