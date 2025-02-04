import torch
from torch.autograd import Variable
import numpy as np
import os
from tensorboardX import SummaryWriter

from tools import train_func as tf
from tools.evaluate_func import svm


class ModelNetTrainer(object):

    def __init__(self, model, device, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, model_dir, num_classes, num_views=12):

        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.num_views = num_views
        self.num_classes = num_classes

        self.model_dir = model_dir
        self.checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        self.tensorboard_dir = os.path.join(model_dir, 'tensorboard')

        # self.model.cuda()
        self.model.to(self.device)

        self.writer = SummaryWriter(self.tensorboard_dir)

    def train(self, n_epochs):

        if self.model_name == 'mvcnn':
            for param in self.model.net_1.parameters():
                param.requires_grad = False

        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):
            # permute data
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            for i, data in enumerate(self.train_loader):

                if self.model_name == 'mvcnn':
                    N,V,C,H,W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).to(self.device)
                else:
                    in_data = Variable(data[1]).to(self.device)
                target = Variable(data[0]).to(self.device).long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)
                loss, [discrimn_loss, compress_loss] = self.loss_fn(out_data, target, num_classes=self.num_classes)
                
                self.writer.add_scalars('train_loss',
                                        {'loss': loss, 'discrimn': discrimn_loss, 'compress': compress_loss},
                                        i_acc+i+1)

                loss.backward()
                self.optimizer.step()

                # for name in self.model.state_dict():
                #     print(name)
                # print(self.model.state_dict()['net_1.6.bias'])
                # print(self.model.state_dict()['net_2.2.9.weight'])
                
                log_str = 'epoch %d, step %d: train_loss %.3f' % (epoch+1, i+1, loss)
                if (i+1) % 1 == 0:
                    print(log_str)
            i_acc += i+1

            # evaluation
            if (epoch+1) % 10 == 0:
                with torch.no_grad():
                    acc_train_svm, acc_test_svm = self.update_validation_accuracy()
                self.writer.add_scalars('acc',
                                        {'acc_train_svm': acc_train_svm, 'acc_test_svm': acc_test_svm},
                                        epoch+1)

            # save the model
            if (epoch + 1) % 10 == 0:
                self.model.save(self.checkpoints_dir, epoch)
 
            # adjust learning rate manually
            if epoch > 0 and (epoch+1) % 40 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.checkpoints_dir+"/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self):

        self.model.eval()
        train_features, train_labels = tf.get_features(self.model, self.train_loader, 'mcr2', self.model_name)
        test_features, test_labels = tf.get_features(self.model, self.val_loader, 'mcr2', self.model_name)

        # train & test accuracy
        acc_train_svm, acc_test_svm = svm(train_features, train_labels, test_features, test_labels)

        self.model.train()

        return acc_train_svm, acc_test_svm

