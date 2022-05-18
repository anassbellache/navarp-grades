import torch.nn as nn

class SrCNN(nn.Module):
    def __init__(self):
        super(SrCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(kernel_size=(3,3), in_channels=1, out_channels=64, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for i in range(15):
            layers.append(nn.Conv2d(kernel_size=(3,3), in_channels=64, out_channels=64, padding=1))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(kernel_size=(3,3), in_channels=64, out_channels=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for i in range(len(layers)):
            if i % 2 == 0:
                nn.init.xavier_uniform_(layers[i].weight)
        
        self.srcnn = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.srcnn(x)
        out = self.dropout(out)
        return out
    


