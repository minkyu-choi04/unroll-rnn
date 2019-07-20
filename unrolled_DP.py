import torch
import torch.optim as optim
import torch.nn as nn
import time
from torch.autograd import Variable
from time import sleep

input_size = 1024
hidden_size = 1024*2
batch_size = 256#512#1024
time_steps = 16
num_layers = 4
lr   = 4e-2

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        
    def forward(self, input):
        outputs = []
        for step in range(time_steps):
            if step == 0:
                hidden = (torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda(), 
                        torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            output, hidden = self.model(torch.unsqueeze(input[:, step, :],1), hidden)
            outputs.append(output)
        return outputs

model = Model(input_size, hidden_size, num_layers)
model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
model.cuda()
optimizer = optim.SGD(model.parameters(),lr = lr,momentum=0.9,dampening = 0.0, weight_decay = 0.0)
criterion = nn.MSELoss().cuda()

input = torch.randn(batch_size, time_steps, input_size).cuda()
target = torch.randn(batch_size, time_steps, hidden_size).cuda()

loss = 0
start = time.time()
for epoch in range(100):
    print('Epoch: {},  Batch: {}'.format(epoch, batch_size))
    loss = 0
    model.zero_grad()
    optimizer.zero_grad()

    outputs = model(input)
    output = torch.cat(outputs, 1)

    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
print("Test ran in " + str( time.time() - start) + " seconds")
