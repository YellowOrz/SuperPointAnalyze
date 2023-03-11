import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGBackbone(torch.nn.Module):
    """vgg backbone to extract feature
    Note:set eps=1e-3 for BatchNorm2d to reproduce results
         of pretrained model `superpoint_bn.pth`
    """

    def __init__(self, config, input_channel=1, device='cpu'):
        super(VGGBackbone, self).__init__()
        self.device = device
        channels = config['channels']

        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, channels[0], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # block 3
        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[6], channels[7], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.block1_1(x)
        out = self.block1_2(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block4_1(out)
        feat_map = self.block4_2(out)
        return feat_map


class VGGBackboneBN(torch.nn.Module):
    """vgg backbone to extract feature
    Note:set eps=1e-3 for BatchNorm2d to reproduce results
         of pretrained model `superpoint_bn.pth`
    """

    def __init__(self, config, input_channel=1, device='cpu'):
        super(VGGBackboneBN, self).__init__()
        self.device = device
        channels = config['channels']

        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, channels[0], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[0]),
        )

        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[1]),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[2]),
        )
        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[3]),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[4]),
        )
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[5]),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # block 3
        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[6]),
        )
        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[6], channels[7], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[7]),
        )

    def forward(self, x):
        out = self.block1_1(x)
        out = self.block1_2(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block4_1(out)
        feat_map = self.block4_2(out)

        return feat_map


def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(r*r), r*H, r*W],
        where r refers to scale factor
    """
    num, ch, height, width = tensor.shape
    assert ch % (scale_factor * scale_factor) == 0

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
    tensor = tensor.permute(0, 1, 4, 2, 5, 3)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor


def box_nms(prob, size=4, iou=0.1, min_prob=0.015, keep_top_k=-1):
    """
    :param prob: probability, torch.tensor, must be [1,H,W]
    :param size: box size for 2d nms
    :param iou:
    :param min_prob:
    :param keep_top_k:
    :return:
    """
    assert (prob.shape[0] == 1 and len(prob.shape) == 3)
    prob = prob.squeeze(dim=0)

    pts = torch.stack(torch.where(prob >= min_prob)).t()
    boxes = torch.cat((pts - size / 2.0, pts + size / 2.0), dim=1).to(torch.float32)
    scores = prob[pts[:, 0], pts[:, 1]]
    indices = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=iou)
    pts = pts[indices, :]
    scores = scores[indices]
    if keep_top_k > 0:
        k = min(scores.shape[0], keep_top_k)
        scores, indices = torch.topk(scores, k)
        pts = pts[indices, :]
    nms_prob = torch.zeros_like(prob)
    nms_prob[pts[:, 0], pts[:, 1]] = scores

    return nms_prob.unsqueeze(dim=0)


class SuperPointBNNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, config, input_channel=1, grid_size=8, device='cpu', using_bn=True):
        super(SuperPointBNNet, self).__init__()
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']
        if using_bn:
            self.backbone = VGGBackboneBN(config['backbone']['vgg'], input_channel, device=device)
        else:
            self.backbone = VGGBackbone(config['backbone']['vgg'], input_channel, device=device)
        ##
        self.detector_head = DetectorHead(input_channel=config['det_head']['feat_in_dim'],
                                          grid_size=grid_size, using_bn=using_bn)
        self.descriptor_head = DescriptorHead(input_channel=config['des_head']['feat_in_dim'],
                                              output_channel=config['des_head']['feat_out_dim'],
                                              grid_size=grid_size, using_bn=using_bn)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        if isinstance(x, dict):
            feat_map = self.backbone(x['img'])
        else:
            feat_map = self.backbone(x)
        det_outputs = self.detector_head(feat_map)

        prob = det_outputs['prob']
        if self.nms is not None:
            prob = [box_nms(p.unsqueeze(dim=0),
                            self.nms,
                            min_prob=self.det_thresh,
                            keep_top_k=self.topk).squeeze(dim=0) for p in prob]
            prob = torch.stack(prob)
            det_outputs.setdefault('prob_nms', prob)

        pred = prob[prob >= self.det_thresh]
        det_outputs.setdefault('pred', pred)

        desc_outputs = self.descriptor_head(feat_map)
        return {'det_info': det_outputs, 'desc_info': desc_outputs}


class DetectorHead(torch.nn.Module):
    def __init__(self, input_channel, grid_size, using_bn=True):
        super(DetectorHead, self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn
        ##
        self.convPa = torch.nn.Conv2d(input_channel, 256, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convPb = torch.nn.Conv2d(256, pow(grid_size, 2) + 1, kernel_size=1, stride=1, padding=0)

        self.bnPa, self.bnPb = None, None
        if using_bn:
            self.bnPa = torch.nn.BatchNorm2d(256)
            self.bnPb = torch.nn.BatchNorm2d(pow(grid_size, 2) + 1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = None
        if self.using_bn:
            out = self.bnPa(self.relu(self.convPa(x)))
            out = self.bnPb(self.convPb(out))  # (B,65,H,W)
        else:
            out = self.relu(self.convPa(x))
            out = self.convPb(out)  # (B,65,H,W)

        prob = self.softmax(out)
        prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        # Reshape to get full resolution heatmap.
        prob = pixel_shuffle(prob, self.grid_size)  # [B,1,H*8,W*8]
        prob = prob.squeeze(dim=1)  # [B,H,W]

        return {'logits': out, 'prob': prob}


class DescriptorHead(torch.nn.Module):
    def __init__(self, input_channel, output_channel, grid_size, using_bn=True):
        super(DescriptorHead, self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn

        self.convDa = torch.nn.Conv2d(input_channel, 256, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convDb = torch.nn.Conv2d(256, output_channel, kernel_size=1, stride=1, padding=0)

        self.bnDa, self.bnDb = None, None
        if using_bn:
            self.bnDa = torch.nn.BatchNorm2d(256)
            self.bnDb = torch.nn.BatchNorm2d(output_channel)

    def forward(self, x):
        out = None
        if self.using_bn:
            out = self.bnDa(self.relu(self.convDa(x)))
            out = self.bnDb(self.convDb(out))
        else:
            out = self.relu(self.convDa(x))
            out = self.convDb(out)

        # out_norm = torch.norm(out, p=2, dim=1)  # Compute the norm.
        # out = out.div(torch.unsqueeze(out_norm, 1))  # Divide by norm to normalize.

        # TODO: here is different with tf.image.resize_bilinear
        desc = F.interpolate(out, scale_factor=self.grid_size, mode='bilinear', align_corners=False)
        desc = F.normalize(desc, p=2, dim=1)  # normalize by channel

        return {'desc_raw': out, 'desc': desc}


if __name__ == '__main__':
    model = SuperPointBNNet()
    model.load_state_dict(torch.load('../ckpt/superpoint_bn.pth'))
    print('Done')
