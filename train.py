import os
import torch
from argparse import ArgumentParser
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.core.utils.helper import load_config, show_images
from src.core.model import Model
from src.core.utils.dataset import ImageDataset

parser = ArgumentParser()

parser.add_argument('-config', help='path to config file')

args = parser.parse_args()

# loading config from yaml file
cfg = load_config(args.config)
device = cfg['device']
if device == 'cpu':
    device = torch.device('cpu')
elif device == 'gpu':
    os.environ['CUDA_AVAILABLE_DEVICES'] = cfg['gpu_num']
    device = torch.device('cuda:0')

# define model
model = Model(cfg=cfg, device=device)
# preprocessing of input images
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# inverse
inverse1 = transforms.Normalize(mean=[0, 0, 0],
                                std=[1/0.229, 1/0.224, 1/0.225])

inverse2 = transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1, 1, 1])

inverse = transforms.Compose([inverse2, inverse1])

totensor = transforms.ToTensor()
transform = transforms.Compose([totensor, normalize])

# define dataset and dataloader
dataset = ImageDataset()

dataloader = DataLoader(dataset=dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                        drop_last=True)


num_epochs = cfg['num_epochs']
log_dir = cfg['log_dir']
writer = SummaryWriter(log_dir=log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
check_dir = cfg['checkpoint_dir']
if not os.path.exists(check_dir):
    os.makedirs(check_dir)

return_imgs = False
# training loop
i = 0
for epoch in range(0, num_epochs):
    for batch in dataloader:
        frame1, frame2 = batch
        frame1 = frame1.to(device)
        frame2 = frame2.to(device)
        if i % 500 == 0 and i != 0:
            return_imgs = True
        values = model.train_step(frame1, frame2, return_imgs)
        # tensorboard logs, show images every 500 iterations
        if not return_imgs:
            writer.add_scalars(main_tag='losses', tag_scalar_dict=values, global_step=i)
        else:
            values_ = {k: v for k, v in values.items() if len(v.shape) == 0}
            writer.add_scalars(main_tag='losses', tag_scalar_dict=values_, global_step=i)
            for k, v in values.items():
                if len(v.shape) > 1:
                    if k == 'transformed_template':
                        grid = show_images(v, renorm=None)
                    else:
                        grid = show_images(v, renorm=inverse)
                    writer.add_image(k, grid, global_step=i)
        # print(i)
        i += 1
        return_imgs = False
    # save checkpoint
    torch.save({'regressor': model.regressor.state_dict(), 'translator': model.translator.state_dict(),
                'optim': model.optim.state_dict()}, os.path.join(check_dir, 'checkpoint_' + str(i) + '.tar'))





