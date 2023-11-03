import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg

class NeRF(nn.Module):
    def __init__(self, output_ch = 4, skips = [4], use_viewdirs = True):
        super(NeRF, self).__init__()
        net_cfg = cfg.network
        self.xyz_encoder, self.xyz_input_ch = get_encoder(cfg.network.xyz_encoder)
        self.dir_encoder, self.dir_input_ch = get_encoder(cfg.network.dir_encoder)
        self.Width = net_cfg.W # Width of full-connected layers
        self.Depth = net_cfg.D # depth of full-connected layers
        self.output_ch = output_ch # output channels alpha + rgb
        self.skips = skips # skip connections
        self.use_viewdirs = use_viewdirs
        # full connect, without view
        self.full_linear = nn.ModuleList(
            [nn.Linear(self.xyz_input_ch, self.Width)] + [nn.Linear(self.Width, self.Width) 
                if i not in skips else nn.Linear(
                    self.Width + self.xyz_input_ch, self.Width
                ) for i in range(self.Depth-1)]
        )
        
        # with view
        self.view_linear = nn.ModuleList(
            [nn.Linear(self.dir_input_ch+ self.Width, self.Width // 2)]
        )

        if use_viewdirs:
            self.alpha_linear = nn.Linear(self.Width, 1)
            self.feature_linear = nn.Linear(self.Width, self.Width)
            self.rgb_linear = nn.Linear(self.Width // 2, 3)  # last layer
        else:
            self.output_linear = nn.Linear(self.Width, output_ch)

        pass
    def net_forward(self, x):   # x: [N_rays * N_samples, 90]
        pts, views = torch.split(x, [self.xyz_input_ch, self.dir_input_ch], dim=-1)
        o = pts
        for i, layer in enumerate(self.full_linear):    # 全连层
            o = self.full_linear[i](o)
            o = F.relu(o)
            if i in self.skips: # 残差连接
                o = torch.cat([pts, o], dim=-1)
        if self.use_viewdirs:
            alpha = self.alpha_linear(o)    # 体密度，只和位置有关
            feature = self.feature_linear(o)
            o = torch.cat([feature, views], dim=-1)
            
            for i, layer in enumerate(self.view_linear):
                o = self.view_linear[i][o]
                o = F.relu(o)   # [N_rays * N_samples, 128]
            
            rgb = self.rgb_linear(o)
            output = torch.cat([rgb, alpha], dim=-1)
        else:
            output = self.output_linear(o)
        return output
    
    def coarse_net(self, rays, viewdirs):
        input = torch.reshape(rays, [-1, rays.shape])   # [N_rays * N_samples, 3]
        embedded = self.xyz_encoder(input)  # 对位置进行编码 [N_rays * N_samples, 63]
        if self.use_viewdirs:
            view_dirs = viewdirs[:, None].expand(rays.shape)
            view_dirs_flat = torch.reshape(view_dirs, [-1, view_dirs.shape[-1]])
            embedded_dirs = self.dir_encoder(view_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)
        return self.net_forward(embedded) 
    pass

    def render_rays(self, rays, N_samples, lindisp=False, perturb = 0.): 
        white_bkgd = cfg.task_arg.white_bkgd    # 是否使用白色背景
        N_rays = rays.shape[0]
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        if rays.shape[1] > 8:   # 如果添加了视角信息
            viewdirs = rays[:, -3:]
        else:
            viewdirs = None
        bounds = torch.reshape(rays[:, 6:8], [-1, 1, 2])
        near = bounds[..., 0]
        far = bounds[..., 1]

        t_vals = torch.linspace(0., 1., steps=N_samples)
        if lindisp:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)  # 线性采样
        else:   # 差值采样  
            z_vals = near * (1. - t_vals) + far * t_vals
        z_vals = z_vals.expand([N_rays, N_samples])
        if perturb > 0.:    # 增添一些随机性   
            mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            lower = torch.cat([z_vals[:, 0], mids], -1)
            upper = torch.cat([mids, z_vals[:, -1]], -1)

            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand
        
        # o + td
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., : , None]   # [N_rays, N_samples, 3]

        coarse_res = self.coarse_net(pts, viewdirs) # get the result from the coarse net [N_rays, N_samples, 4]
        





        pass
    def batchify(self, rays):   # 控制光线数量
        chunk_size = cfg.task_arg.chunk_size
        N_samples = cfg.task_arg.cascade_samples[0] # N_samples = 64
        all_ret = {}
        for i in range(0, rays.shape[0], chunk_size):
            ret = self.render_rays(rays[i:i+chunk_size], N_samples)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
            pass
        
        pass
    def forward(self, batch):
        use_viewdirs = self.use_viewdirs
        rays_o, rays_d = batch['rays_o'], batch['rays_d']

        if use_viewdirs:
            viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)    # 规范化
            viewdirs = torch.reshape(viewdirs, [-1, 3], float())    # [N_rays, 3]

        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near = near * torch.ones(rays_d.shape[0], device=rays_d.device)
        far = far * torch.ones(rays_d.shape[0], device=rays_d.device)

        rays = torch.cat([rays_o, rays_d, near[:, None], far[:, None]])    # [N_rays, 8]
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], 1)    # [N_rays, 11] 如果要加上视角的话
        
        pass