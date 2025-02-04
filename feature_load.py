import numpy as np
import torch
import argparse

from tools.img_dataset import MultiviewImgDataset
from models.mvcnn import MVCNN, SVCNN

from tools import train_func as tf

parser = argparse.ArgumentParser()
parser.add_argument("--bs", "--batch_size", type=int, default=30,
                    help="batch size")
parser.add_argument("--cnn_name", type=str, default="vgg11",
                    help="cnn model name")
parser.add_argument("--num_views", type=int, default=3,
                    help="number of views")
parser.add_argument("--num_classes", type=int, default=10,
                    help="number of classes")
parser.add_argument("--train_path", type=str, default="modelnet40_images_new_12x/*/train")
parser.add_argument("--val_path", type=str, default="modelnet40_images_new_12x/*/test")
parser.add_argument("--num_workers", type=int, default=24,
                    help="number of workers")
parser.add_argument('--svcnn_model_dir', type=str, default='./mvcnn/phase1_classes10_views3_fd1_32_bs1200_lr0.0001_wd0.001_eps0.5/checkpoints/svcnn/model-00029.pth',
                    help='base directory for svcnn model')
parser.add_argument('--mvcnn_model_dir', type=str, default='./mvcnn/phase2_classes10_views3_fd1_32_fd2_8_bs1200_lr0.0001_wd0.001_eps0.5/checkpoints/mvcnn/model-00199.pth',
                    help='base directory for mvcnn model')
parser.add_argument('--pretraining', type=bool, default=True,
                    help='pretraining')
parser.add_argument('--fd_phase1', type=int, default=32,
                    help='dimension of feature dimension in phase 1')
parser.add_argument('--fd_phase2', type=int, default=8,
                    help='dimension of feature dimension per user in phase 2')
args = parser.parse_args()

## CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    cnet = SVCNN(name='svcnn', nclasses=args.num_classes, pretraining=args.pretraining,
                 cnn_name=args.cnn_name, fd_phase1=args.fd_phase1)
    state_dict_1 = torch.load(args.svcnn_model_dir, map_location=torch.device('cpu'))
    cnet.load_state_dict(state_dict_1)
    cnet_2 = MVCNN(name='mvcnn', model=cnet, nclasses=args.num_classes, cnn_name=args.cnn_name,
                   num_views=args.num_views, fd_phase1=args.fd_phase1, fd_per_user=args.fd_phase2)
    state_dict_2 = torch.load(args.mvcnn_model_dir, map_location=torch.device('cpu'))
    cnet_2.load_state_dict(state_dict_2)
    cnet_2.eval()
    del cnet

    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False,
                                        num_classes=args.num_classes, num_views=args.num_views, train_objects=9999)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                                               shuffle=False, num_workers=args.num_workers)
    # shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True,
                                      num_classes=args.num_classes, num_views=args.num_views, test_objects=9999)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs,
                                             shuffle=False, num_workers=args.num_workers)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))

    train_features, train_labels = tf.get_features(cnet_2, train_loader, 'mcr2', 'mvcnn')
    feature = np.array(train_features.detach().cpu())
    target = np.array(train_labels.detach().cpu())
    mdic = {"train_feature": feature, "train_label": target}
    # savemat(f"MCR2_ModelNet10_train_feature_label_36.mat", mdic)

    test_features, test_labels = tf.get_features(cnet_2, val_loader, 'mcr2', 'mvcnn')
    feature = np.array(test_features.detach().cpu())
    target = np.array(test_labels.detach().cpu())
    mdic = {"test_feature": feature, "test_label": target}
    # savemat(f"MCR2_ModelNet10_test_feature_label_36.mat", mdic)


