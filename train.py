import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataset import *
from torchsummary import summary
from models import fastvit
import matplotlib.pyplot as plt
# For reproducibility
torch.manual_seed(1)

# custom_datasets = CustomImageDataset(data, label)



if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    numSample_list = [636, 519]
    weights = [1 - (x / sum(numSample_list)) for x in numSample_list]
    weights = torch.FloatTensor(weights).to(device)
    
    ep = 0
    ls = 1
    lr = 3*1e-4
    epochs = 100
    batch_size = 128
    freezing = False
    
    loss_ = []
    
    
    model = fastvit.egotouch().to(device)
    
    model.load_state_dict(torch.load("fastvit_t8.pth.tar"),strict=False)
    
    force_criterion = nn.MSELoss()
    touch_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)
    trainsets = CustomImageDataset('dataset', 'train')
    trainloader = DataLoader(trainsets,batch_size, shuffle=True)
    
    n = len(trainloader)
    # summary(model, (3, 224, 224), batch_size=128)
    
    
    if freezing :
        for name, param in model.named_parameters():
            if name.count("feature"):
                print(name)
                param.requires_grad = False
        
    
    # for epoch in range(epochs):
    #     running_loss = 0.0
    #     for data in trainloader:
    #         inputs, r_theta, values = data

    #         values_force, values_touch = torch.chunk(values,2, dim = 1)
            
    #         optimizer.zero_grad()
    #         output_force, output_touch = model(inputs.to(device), r_theta.to(device))  
    #         print(output_touch.shape) 
    #         loss = force_criterion(output_force, values_force.to(device))* 5  + touch_criterion(output_touch, values_touch.to(device))
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
        
    #     l = running_loss / n
    #     loss_.append(l)
        
    #     if l < ls:
    #         ep = epoch
    #         torch.save({'epoch': ep,
    #                 'loss': loss_,
    #                 'model': model.state_dict(),
    #                 'optimizer': optimizer.state_dict()
    #                 }, './models/egotouch.pt')
    # print('Finished Training')

    plt.figure(figsize=(10, 8))
    plt.plot(loss_)
    plt.title("Training Loss")
    plt.xlabel("epoch")
    plt.show()
    plt.savefig('loss_graph.jpg')

    checkpoint = torch.load('./models/egotouch.pt')
    model.load_state_dict(checkpoint['model']) 
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_ = checkpoint['loss']
    ep = checkpoint['epoch']
    ls = loss_[-1]
    print(f"epoch={ep}, loss={ls}")