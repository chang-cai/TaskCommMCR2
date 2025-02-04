import torch
import torch.optim as optim
import os
import argparse

from tools.trainer import ModelNetTrainer
from tools.img_dataset import MultiviewImgDataset
from models.mvcnn import MVCNN, SVCNN

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
parser.add_argument('--svcnn_model_dir', type=str, default='./mvcnn/phase1_classes10_views3_fd1_32_bs1200_lr0.0001_wd0.001_eps0.5/checkpoints/svcnn/model-00029.pth',
                    help='base directory for svcnn model')
parser.add_argument('--epoch', type=int, default=500,
                    help='number of epochs for training')
parser.add_argument('--eps', type=float, default=0.5,
                    help='eps squared')
parser.add_argument('--fd_phase1', type=int, default=32,
                    help='dimension of feature dimension in phase 1')
parser.add_argument('--fd_phase2', type=int, default=8,
                    help='dimension of feature dimension per user in phase 2')
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
                             'phase2_classes{}_views{}_fd1_{}_fd2_{}_bs{}_lr{}_wd{}_eps{}{}'.format(
                                 args.num_classes, args.num_views, args.fd_phase1, args.fd_phase2, args.bs, args.lr,
                                 args.weight_decay, args.eps, args.tail))
    utils.init_pipeline(model_dir)
    utils.save_params(model_dir, vars(args))

    cnet = SVCNN(name='svcnn', nclasses=args.num_classes, pretraining=args.pretraining,
                 cnn_name=args.cnn_name, fd_phase1=args.fd_phase1)
    state_dict = torch.load(args.svcnn_model_dir, map_location=torch.device('cpu'))
    cnet.load_state_dict(state_dict)
    cnet_2 = MVCNN(name='mvcnn', model=cnet, nclasses=args.num_classes, cnn_name=args.cnn_name,
                   num_views=args.num_views, fd_phase1=args.fd_phase1, fd_per_user=args.fd_phase2)
    del cnet

    cnet_2 = cnet_2.to(device)
    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False,
                                     num_classes=args.num_classes, num_views=args.num_views, train_objects=9999)  # 80
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                                               shuffle=False, num_workers=args.num_workers)
    # shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True,
                                   num_classes=args.num_classes, num_views=args.num_views, test_objects=9999)  # 20
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs,
                                             shuffle=False, num_workers=args.num_workers)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))

    loss_fn = MaximalCodingRateReduction(gam1=1, gam2=1, eps=args.eps)
    trainer = ModelNetTrainer(cnet_2, device, train_loader, val_loader, optimizer, loss_fn, 'mvcnn',
                              model_dir, num_classes=args.num_classes, num_views=args.num_views)
    trainer.train(args.epoch)


