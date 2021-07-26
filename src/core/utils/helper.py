import yaml
import torch
import cv2
import json_tricks as json
from torchvision.utils import make_grid


def load_config(path):
    with open(path, 'r') as file:
        cfg = yaml.load(file)

    return cfg


def draw_shape(pos, sigma_x, sigma_y, angle, size):
    """
    draw (batched) gaussian with sigma_x, sigma_y on 2d grid

    Args:
        pos: torch.tensor (float) with shape (2) specifying center of gaussian blob
        sigma_x: torch.tensor (float scalar), scaling parameter along x-axis
        sigma_y: similar along y-axis
        angle: torch.tensor (float scalar) rotation angle in radians
        size: int specifying size of image
        device: torch.device, either cpu or gpu

    Returns:
        torch.tensor (1, 1, size, size) with gaussian blob
    """
    device = pos.device
    assert sigma_x.device == sigma_y.device == angle.device == device, "inputs should be on the same device!"

    # create 2d meshgrid
    x, y = torch.meshgrid(torch.arange(0, size), torch.arange(0, size))
    x, y = x.unsqueeze(0).unsqueeze(0).to(device), y.unsqueeze(0).unsqueeze(0).to(device)

    # see https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    a = torch.cos(angle) ** 2 / (2 * sigma_x ** 2) + torch.sin(angle) ** 2 / (2 * sigma_y ** 2)
    b = -torch.sin(2 * angle) / (4 * sigma_x ** 2) + torch.sin(2 * angle) / (4 * sigma_y ** 2)
    c = torch.sin(angle) ** 2 / (2 * sigma_x ** 2) + torch.cos(angle) ** 2 / (2 * sigma_y ** 2)

    # append dimsensions for broadcasting
    pos = pos.view(1, 1, 2, 1, 1)
    a, b, c = a.view(1, 1), b.view(1, 1), c.view(1, 1)

    # pixel-wise distance from center
    xdist = (x - pos[:, :, 0])
    ydist = (y - pos[:, :, 1])

    # gaussian function
    g = torch.exp((-a * xdist ** 2 - 2 * b * xdist * ydist - c * ydist ** 2))

    return g


def draw_template(path, size, batch_size, device):
    """
    draw template consisting of limbs defined by gaussian heatmap
    Args:
        template: json file defining all parts
        size: int, image size (assumed quadratic), this should match the center coordinates defined in the json!
        device: torch.device, either cpu or gpu
    """
    with open(path, 'r') as file:
        template = json.load(file)
    heatmaps = []
    for v in template.values():
        center = torch.tensor(v['center']).to(device)
        sx = torch.tensor(v['sx']).to(device)
        sy = torch.tensor(v['sy']).to(device)
        angle = torch.tensor(v['angle']).to(device)
        heatmaps.append(draw_shape(center, sx, sy, angle, size))

    heatmaps = torch.cat(heatmaps, dim=1).repeat(batch_size, 1, 1, 1)

    return heatmaps


def load_anchor_points(path, device, batch_size):
    """
    load anchor points from json file
    Args:
        anchor_points: json file containing anchor points per part in col, row format similar to open-cv
        device: torch.device, either cpu or gpu
    """
    with open(path, 'r') as file:
        anchor_points = json.load(file)
    # assumes three anchor points for core, two (parent+child) for all others except hands and feet and head
    # change this accordingly for different template definitions!
    double = []
    single = []
    for k, v in anchor_points.items():
        if k in ['left hand', 'right hand', 'left foot', 'right foot', 'head']:
            single.append(v)
        elif k == 'core':
            triple = [v]
        else:
            double.append(v)

    return torch.tensor(triple).to(device).float().unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1), \
           torch.tensor(single).to(device).float().unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1), \
           torch.tensor(double).to(device).float().unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)


def show_images(tensor, renorm):
    if renorm:
        for i in range(tensor.shape[0]):
            #  bgr opencv to rgb
            tensor[i] = renorm(tensor[i])[[2, 1, 0]]
    output_grid = make_grid(tensor, nrow=6, normalize=True, scale_each=True)
    return output_grid