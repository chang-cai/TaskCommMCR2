import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .model import Model
import copy


# mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
# std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11', fd_phase1=32):
        super(SVCNN, self).__init__(name)

        if nclasses == 10:
            self.classnames = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
                               'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        elif nclasses == 40:
            self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                               'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                               'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                               'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                               'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        # self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        # self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False)
        self.fd_phase1 = fd_phase1

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                # self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
                self.net_2 = torch.nn.Sequential(
                    nn.Linear(25088, 4096, bias=True),  # False
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, self.fd_phase1, bias=True)
                )
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            # self.net_2._modules['6'] = nn.Linear(4096, self.fd)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            feature = F.normalize(self.net_2(y.view(y.shape[0], -1)))
            return feature


class MVCNN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12, fd_phase1=32, fd_per_user=6):
        super(MVCNN, self).__init__(name)

        if nclasses == 10:
            self.classnames = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
                               'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        elif nclasses == 40:
            self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                               'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                               'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                               'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                               'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.fd_per_user = fd_per_user
        self.fd_phase1 = fd_phase1
        # self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        # self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False)

        # self.use_resnet = cnn_name.startswith('resnet')
        #
        # if self.use_resnet:
        #     self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
        #     self.net_2 = model.net.fc
        # else:
        #     self.net_1 = model.net_1
        #     self.net_2 = model.net_2

        self.net_1 = model.net_1

        net_2_list = []
        for _ in range(self.num_views):
            net_2 = copy.deepcopy(model.net_2)
            net_2._modules['4'] = nn.BatchNorm1d(self.fd_phase1)
            net_2._modules['5'] = nn.ReLU(inplace=True)
            net_2._modules['6'] = nn.Linear(self.fd_phase1, 32, bias=True)  # False
            net_2._modules['7'] = nn.BatchNorm1d(32)
            net_2._modules['8'] = nn.ReLU(inplace=True)
            net_2._modules['9'] = nn.Linear(32, self.fd_per_user, bias=True)

            net_2_list.append(net_2)
        self.net_2 = nn.ModuleList(net_2_list)

    def forward(self, x):
        y = self.net_1(x)  # (bs*views,512,7,7)
        y = y.view((int(x.shape[0] / self.num_views), self.num_views, -1))  # (bs,views,25088)

        feature_list = []
        for i in range(self.num_views):
            feature_i = self.net_2[i](y[:, i])  # (bs, fd_i)
            feature_list.append(feature_i)

        feature = torch.cat(feature_list, dim=1)  # (bs, fd_i*views)
        return F.normalize(feature)
