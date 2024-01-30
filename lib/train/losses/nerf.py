import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.img2mse = lambda x, y: torch.mean((x - y) ** 2)
        self.mse2psnr = lambda x : -10. * torch.log(x.detach()) / torch.log(torch.Tensor([10.]).to(x.device))
    
    def forward(self, batch):
        image_stats = {}
        scalar_stats = {}

        output = self.net(batch)
        rgb_res = output['rgb_map']
        targets = batch['target']
        loss = self.img2mse(rgb_res, targets)
        psnr = self.mse2psnr(loss)
        if 'rgb0' in output:    # if used two nets
            loss0 = self.img2mse(output['rgb0'], targets)
            loss += loss0
            psnr0 = self.mse2psnr(loss0)
        
        scalar_stats.update({'loss': loss})
        scalar_stats.update({'psnr': psnr})

        return output, loss, scalar_stats, image_stats
        



