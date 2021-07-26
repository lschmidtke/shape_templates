import torch.nn as nn
import torch

l2_loss = nn.MSELoss()

l1_loss = nn.L1Loss()


def _l1_loss(x1, x2):
    return l1_loss(x1, x2)


def _l2_loss(x1, x2):
    return l2_loss(x1, x2)


def compute_anchor_loss1(core, double, single, size):
    """
    compute mean distance between pairs of transformed anchor_points,
    change this according to your connectivity constraints
    """
    loss = 0
    # normalize to range 0, 1
    core = core/size
    single = single/size
    double = double/size

    # loss between core and hips and shoulders
    indices1 = [0, 0, 1, 1]
    # hips and shoulders
    indices2 = [0, 1, 6, 7]

    print(core[:, 0, indices1].shape)
    print(double[:, indices2, 0].shape)

    loss += (core[:, 0, indices1] - double[:, indices2, 0]).sum(1).pow(2).mean()

    # head and core
    # loss += l2_loss(core[:, 0, -1], single[:, -1, 0])

    loss += (core[:, 0, -1] - single[:, -1, 0]).sum(1).pow(2).mean()

    # hips to thighs to shins, shoulders to arms to forearms
    indices3 = [0, 1, 2, 3, 6, 7, 8, 9]
    indices4 = [2, 3, 4, 5, 8, 9, 10, 11]
    #
    # loss += l2_loss(double[:, indices3, 1], double[:, indices4, 0])

    loss += (double[:, indices3, 1] - double[:, indices4, 0]).sum(1).pow(2).mean()

    #  shin to feet, forarms to hands
    indices5 = [4, 5, 10, 11]
    indices6 = [0, 1, 2, 3]

    # loss += l2_loss(double[:, indices5, 1], single[:, indices6, 0])

    loss += (double[:, indices5, 1] - single[:, indices6, 0]).sum(1).pow(2).mean()

    return loss


def compute_anchor_loss2(core, double, single, size):
    """
    compute mean distance between pairs of transformed anchor_points,
    change this according to your connectivity constraints
    """
    loss = 0
    # normalize to range 0, 1
    core = core/size
    single = single/size
    double = double/size

    # loss between core and hips and shoulders
    indices1 = [0, 0, 1, 1]
    # hips and shoulders
    indices2 = [0, 1, 6, 7]

    for index1, index2 in zip(indices1, indices2):

        loss += l2_loss(core[:, 0, index1], double[:, index2, 0])
    # head and core
    # loss += l2_loss(core[:, 0, -1], single[:, -1, 0])

    # loss += (core[:, 0, -1] - single[:, -1, 0]).sum(1).pow(2).sqrt().mean()

    loss += l2_loss(core[:, 0, -1], single[:, -1, 0])

    # hips to thighs to shins, shoulders to arms to forearms
    indices3 = [0, 1, 2, 3, 6, 7, 8, 9]
    indices4 = [2, 3, 4, 5, 8, 9, 10, 11]

    for index3, index4 in zip(indices3, indices4):
        loss += l2_loss(double[:, index3, 1], double[:, index4, 0])

    #  shin to feet, forarms to hands
    indices5 = [4, 5, 10, 11]
    indices6 = [0, 1, 2, 3]

    # loss += l2_loss(double[:, indices5, 1], single[:, indices6, 0])

    for index5, index6 in zip(indices5, indices6):

        loss += l2_loss(double[:, index5, 1], single[:, index6, 0])

    return loss


threshold = nn.Threshold(1, 0)


def compute_boundary_loss(core, single, double, img_size):
    """
    compute boundary loss, boundaries are 0 and 1
    loss = x if x smaller or greater than 0, 1
    0 otherwise
    """
    core = core.view(core.shape[0], -1, core.shape[3])
    single = core.view(single.shape[0], -1, single.shape[3])
    double = core.view(double.shape[0], -1, double.shape[3])

    comb = torch.cat([core, single, double], dim=1)

    # normalize to range -1  to 1
    comb = (comb / img_size) * 2 - 1

    return threshold(torch.abs(comb)).sum(1).mean()

