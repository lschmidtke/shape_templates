import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.models as models
from src.core.networks import ParameterRegressor, ImageTranslator
from src.core.utils.losses import compute_anchor_loss, compute_boundary_loss, _l1_loss
from src.core.utils.helper import draw_template, load_anchor_points
from src.core.utils.transforms import transform_template, transform_anchor_points


class Model:
    def __init__(self, cfg, device):

        self.template = draw_template(cfg['template_path'], size=cfg['img_size'], batch_size=cfg['batch_size'],
                                      device=device)
        self.core, self.single, self.double = load_anchor_points(cfg['anchor_pts_path'], device, cfg['batch_size'])
        self.regressor = ParameterRegressor(num_features=cfg['regressor_nf'], num_parts=cfg['num_parts']).to(device)
        self.translator = ImageTranslator(num_features=cfg['translator_nf'], num_parts=cfg['num_parts']).to(device)

        self.optim = Adam(list(self.regressor.parameters()) + list(self.translator.parameters()),
                          lr=cfg['learning_rate'])
        self.vgg = nn.Sequential()
        vgg = models.vgg19(pretrained=True).features.eval().to(device)

        depth = 14

        for i in range(depth):
            self.vgg.add_module(str(i), vgg[i])

        self.I = torch.eye(3)[0:2].view(1, 1, 2, 3).repeat(cfg['batch_size'], cfg['num_parts'], 1, 1).to(device)
        self.aug = torch.Tensor([0, 0, 1]).view(1, 1, 1, 3).repeat(cfg['batch_size'], cfg['num_parts'], 1, 1).to(device)
        self.lambda1 = cfg['anchor_loss_weight']
        self.labmda2 = cfg['boundary_loss_weight']

    def train_step(self, frame1, frame2, return_imgs=False):
        batch_size = frame1.shape[0]
        num_parts = self.template.shape[1]
        img_size = frame1.shape[2]

        estimated_params = self.regressor(frame1)
        estimated_params = self.I + estimated_params

        # (batch, num_parts) --> (batch*num_parts)
        batched_template = self.template.view(-1, img_size, img_size).unsqueeze(1)
        batched_params = estimated_params.view(-1, 2, 3)
        transformed_template = transform_template(batched_template, batched_params)
        # (batch*num_parts) --> (batch, num_parts)
        transformed_template = transformed_template.view(batch_size, num_parts, img_size, img_size)

        # append [0, 0, 1] as last row to matrices
        A = torch.cat([estimated_params, self.aug], dim=-2)
        transformed_anchors = transform_anchor_points(A, self.core, self.double, self.single)

        reconstructed = self.translator(frame2, transformed_template)
        anchor_loss = compute_anchor_loss(*transformed_anchors, size=img_size)
        boundary_loss = compute_boundary_loss(*transformed_anchors, img_size=img_size)
        recon_loss = _l1_loss(self.vgg(frame1), self.vgg(reconstructed))

        # overall loss, gradient and weight update
        loss = recon_loss + self.lambda1 * anchor_loss + self.labmda2 * boundary_loss
        # reset gradients
        self.regressor.zero_grad()
        self.translator.zero_grad()
        loss.backward()
        self.optim.step()

        d = {'anchor_loss': anchor_loss, 'boundary_loss': boundary_loss, 'recon_loss': recon_loss}
        if return_imgs:
            d['reconstructed_frame'] = reconstructed.detach()
            d['frame1'] = frame1
            d['frame2'] = frame2
            d['transformed_template'] = transformed_template.sum(1, keepdims=True).detach()

        return d