import torch
import torch.optim as optim
import torch.nn as nn
import time
from torch.autograd import Variable
from time import sleep
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import os 

input_size = 1024
hidden_size = 1024*2
batch_size = 512#1024
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

def main():
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node * 1
    mp.spawn(main_worker, nprocs=ngpus_per_node, 
            args=(ngpus_per_node, world_size))


def main_worker(gpu, ngpus_per_node, world_size):
    torch.cuda.set_device(gpu)
    print("use GPU: {} for training".format(gpu))
    
    dist.init_process_group(backend='nccl', 
                            init_method='tcp://127.0.0.1:23456',
                            world_size=world_size,  
                            rank=gpu)

    model = Model(input_size, hidden_size, num_layers)
    model.cuda(gpu)
    model = DistributedDataParallel(model, device_ids=[gpu])
    criterion = nn.MSELoss().cuda(gpu)
    optimizer = optim.SGD(model.parameters(),lr = lr,momentum=0.9,dampening = 0.0, weight_decay = 0.0)
    
    input = torch.randn(int(batch_size/ngpus_per_node), time_steps, input_size).cuda(gpu)
    target = torch.randn(int(batch_size/ngpus_per_node), time_steps, hidden_size).cuda(gpu)

    loss = 0
    start = time.time()
    for epoch in range(100):
        print('Epoch: {},  GPU: {},  Batch/Total Batch : {}/{}'.format(epoch, gpu, int(batch_size/ngpus_per_node), batch_size))
        loss = 0
        model.zero_grad()
        optimizer.zero_grad()

        outputs = model(input)
        output = torch.cat(outputs, 1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print("Test ran in " + str( time.time() - start) + " seconds")

if __name__ == '__main__':
    main()
