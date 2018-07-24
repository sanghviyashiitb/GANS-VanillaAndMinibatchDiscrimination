import torch
import torch.nn as nn
import torch.nn.functional as  F
from torch.autograd import Variable
from minibatch_discrimination import MiniBatchDiscrimination


BATCH_SIZE = 100
INPUT_DIM = 784




# Instantiate a MiniBatch Discrimination Layer and check output size
mbd_test = MiniBatchDiscrimination(128, 64, 50, BATCH_SIZE)
in_test = torch.randn(BATCH_SIZE,128)
print(in_test.size())
out_test = mbd_test(in_test)
print(out_test.size())


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.lin1 = nn.Linear(INPUT_DIM, 128)
        self.mbd1 = MiniBatchDiscrimination(128, 64, 50, BATCH_SIZE)
        self.lin2 = nn.Linear(192, 1)

    def forward(self, x):
        x = F.leaky_relu(self.lin1(x),0.1)        
        x = F.sigmoid(self.lin2( torch.cat((x, self.mbd1(x)),dim=1) ))
        # x = self.mbd1(x)
        return x

dis1 = Discriminator()
in1 = torch.randn(BATCH_SIZE,INPUT_DIM)
out1 = dis1(in1)
print(out1.size())
