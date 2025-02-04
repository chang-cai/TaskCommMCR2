import torch
import torch.optim as optim
import os
import argparse

from tools.trainer import ModelNetTrainer
from tools.img_dataset import SingleImgDataset
from models.mvcnn import SVCNN

from mcr2_loss import MaximalCodingRateReduction
from tools import utils

parser = argparse.ArgumentParser()
parser.add_argument("--bs", "--batch_size", type=int, default=1200,
                    help="batch size")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate")
parser.add_argument("--weight_decay", type=float, help="weight decay", default=0.001)
parser.add_argument("--cnn_name", type=str, default="vgg11",
                    help="cnn model name")
parser.add_argument("--num_views", type=int, default=3,
                    help="number of views")
parser.add_argument("--num_classes", type=int, default=10,
                    help="number of classes")
parser.add_argument("--train_path", type=str, default="modelnet40_images_new_12x/*/train")
parser.add_argument("--val_path", type=str, default="modelnet40_images_new_12x/*/test")
parser.add_argument("--num_workers", type=int, default=32,
                    help="number of workers")
parser.add_argument('--save_dir', type=str, default='./mvcnn/',
                    help='base directory for saving PyTorch model')
parser.add_argument('--epoch', type=int, default=500,
                    help='number of epochs for training')
parser.add_argument('--eps', type=float, default=0.5,
                    help='eps squared')
parser.add_argument('--fd_phase1', type=int, default=32,
                    help='dimension of feature dimension')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--pretraining', type=bool, default=True,
                    help='pretraining')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum')
args = parser.parse_args()

## CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    ## Pipelines Setup
    model_dir = os.path.join(args.save_dir,
                             'phase1_classes{}_views{}_fd1_{}_bs{}_lr{}_wd{}_eps{}{}'.format(
                                 args.num_classes, args.num_views, args.fd_phase1, args.bs, args.lr,
                                 args.weight_decay, args.eps, args.tail))
    utils.init_pipeline(model_dir)
    utils.save_params(model_dir, vars(args))

    cnet = SVCNN(name='svcnn', nclasses=args.num_classes, pretraining=args.pretraining,
                 cnn_name=args.cnn_name, fd_phase1=args.fd_phase1)
    cnet = cnet.to(device)
    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(cnet.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)

    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False,
                                     num_classes=args.num_classes, num_views=args.num_views, train_objects=9999)  # 80
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                                               shuffle=True, num_workers=args.num_workers)

    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True,
                                   num_classes=args.num_classes, num_views=args.num_views, test_objects=9999)  # 20
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs,
                                             shuffle=False, num_workers=args.num_workers)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))

    loss_fn = MaximalCodingRateReduction(gam1=1, gam2=1, eps=args.eps)
    trainer = ModelNetTrainer(cnet, device, train_loader, val_loader, optimizer, loss_fn, 'svcnn',
                              model_dir, num_classes=args.num_classes, num_views=1)
    trainer.train(args.epoch)


