from __future__ import print_function

import copy
from tkinter import X

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()

        self.k = k

        # Each layer has batchnorm and relu on it
        #TODO
        # layer 1: k -> 64
        self.conv1 = nn.Sequential(nn.Conv1d(self.k, 64, 1), 
            nn.BatchNorm1d(64), 
            nn.ReLU(True))
        #TODO
        # layer 2:  64 -> 128
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(True))
        #TODO
        # layer 3: 128 -> 1024
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), 
            nn.BatchNorm1d(1024), 
            nn.ReLU(True))
        #TODO
        # fc 1024 -> 512
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(True))
        #TODO
        # fc 512 -> 256
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        #TODO
        # fc 256 -> k*k (no batchnorm, no relu)
        self.fc3 = nn.Sequential(
            nn.Linear(256, self.k*self.k)
        )
        #TODO
        # ReLU activationfunction
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, num_points = x.shape
        #TODO
        # apply layer 1
        x = self.conv1(x)
        #TODO
        # apply layer 2
        x = self.conv2(x)
        #TODO
        # apply layer 3
        x = self.conv3(x)
        #TODO
        # do maxpooling and flatten
        pool =  nn.MaxPool1d(x.size(-1))(x)
        flat = nn.Flatten(1)(pool)
        #TODO
        # apply fc layer 1
        x = self.fc1(flat)
        #TODO
        # apply fc layer 2
        x = self.fc2(x)
        #TODO
        # apply fc layer 3
        x = self.fc3(x)
        #TODO
        #reshape output to a b*k*k tensor
        x.view(batch_size, self.k, self.k)
        #TODO
        # define an identity matrix to add to the output. This will help with the stability of the results since we want our transformations to be close to identity
        identity_matrix = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batch_size,1)
        if x.is_cuda:
            identity_matrix = identity_matrix.cuda()
        
        


        #TODO
        # return output
        x = x + identity_matrix
        x = x.view(-1, self.k, self.k)
        return x



class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = True):
        super(PointNetfeat, self).__init__()

        self.feature_transform= feature_transform
        self.global_feat = global_feat

        #TODO
        # Use TNet to apply transformation on input and multiply the input points with the transformation
        self.tnet = TNet(3)
        #TODO
        # layer 1:3 -> 64
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), 
            nn.BatchNorm1d(64), 
            nn.ReLU(True))
        #TODO
        # Use TNet to apply transformation on features and multiply the input features with the transformation 
        #                                                                        (if feature_transform is true)
        if self.feature_transform:
            self.tnet2 = TNet(k=64)
        #TODO
        # layer2: 64 -> 128
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(True))

        #TODO
        # layer 3: 128 -> 1024 (no relu)
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), 
            nn.BatchNorm1d(1024))
        #TODO
        # ReLU activation
        self.relu = nn.ReLU()




    def forward(self, x):
        batch_size, _, num_points = x.shape
        #TODO
        # input transformation, you will need to return the transformation matrix as you will need it for the regularization loss
        transform = self.tnet(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, transform)
        x = x.transpose(2,1)
        #TODO
        # apply layer 1
        x = self.conv1(x)
        #TODO
        # feature transformation, you will need to return the transformation matrix as you will need it for the regularization loss
        if self.feature_transform:
            transform2 = self.tnet2(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, transform2)
            x = x.transpose(2,1)
        else:
            transform2 = None
        #TODO
        # apply layer 2
        pointfeat = x
        x = self.conv2(x)
        #TODO
        # apply layer 3
        x = self.conv3(x)
        #TODO
        # apply maxpooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        #TODO
        # return output, input transformation matrix, feature transformation matrix
        if self.global_feat: # This shows if we're doing classification or segmentation
            
                return x, transform, transform2  
        else:
    
                x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
                return torch.cat([x, pointfeat], 1), transform, transform2


class PointNetCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        #TODO
        # get global features + point features from PointNetfeat
        self.num_classes = num_classes
        self.feature_transform = feature_transform
        self.pointfeat = PointNetfeat(global_feat=False, feature_transform=feature_transform)

        #TODO
        # layer 1: 1088 -> 512
        self.conv1 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        #TODO
        # layer 2: 512 -> 256
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        #TODO
        # layer 3: 256 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        #TODO
        # layer 4:  128 -> k (no ru and batch norm)
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, self.num_classes, 1)
        )

        #TODO
        # ReLU activation
        self.reul = nn.ReLU()

    
    def forward(self, x):
        #TODO
        # You will need these extra outputs: 
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        # (you can directly get them from PointNetfeat)
        batch_size, _, num_points = x.shape
        x, trans, trans_feat = self.pointfeat(x)

        #TODO
        # apply layer 1
        x= self.conv1(x)
        #TODO
        # apply layer 2
        x= self.conv2(x)
        #TODO
        # apply layer 3
        x= self.conv3(x)
        #TODO
        # apply layer 4
        x= self.conv4(x)
        #TODO
        # apply log-softmax
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.num_classes), dim=-1)
        
        x = x.view(batch_size, num_points, self.num_classes)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):

    batch_size, feature_size, _ = trans.shape
    #TODO
    # compute I - AA^t
    I = torch.eye(feature_size)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    #TODO
    # compute norm
    

    #TODO
    # compute mean norms and return
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


if __name__ == '__main__':
    
    sim_data = Variable(torch.rand(32,3,2500))
    trans = TNet(k=3)
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNet(k=64)
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(num_classes = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(num_classes = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())

