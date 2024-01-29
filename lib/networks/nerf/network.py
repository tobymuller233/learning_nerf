import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeRF_network(nn.Module):
    def __init__(self, input_ch = 3, input_ch_views = 3, output_ch = 4, skips = [4], use_viewdirs = True):
        super(NeRF_network, self).__init__()
        self.net_cfg = cfg.network
        self.xyz_encoder, self.xyz_input_ch = get_encoder(cfg.network.xyz_encoder)
        self.dir_encoder, self.dir_input_ch = get_encoder(cfg.network.dir_encoder)
        self.Depth = self.net_cfg.nerf.D # depth of full-connected layers
        self.Width = self.net_cfg.nerf.W # Width of full-connected layers
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
            self.output_linear = nn.Linear(self.Width, self.output_ch)
        
    def forward(self, x):   # x: [N_rays * N_samples, 90]
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
                o = self.view_linear[i](o)
                o = F.relu(o)   # [N_rays * N_samples, 128]
            
            rgb = self.rgb_linear(o)
            output = torch.cat([rgb, alpha], dim=-1)
        else:
            output = self.output_linear(o)
        return output
pass

class Network(nn.Module):
    def __init__(self, output_ch = 4, skips = [4], use_viewdirs = True):
        super(Network, self).__init__()
        self.net_cfg = cfg
        self.xyz_encoder, self.xyz_input_ch = get_encoder(cfg.network.xyz_encoder)
        self.dir_encoder, self.dir_input_ch = get_encoder(cfg.network.dir_encoder)
        self.Width = self.net_cfg.network.nerf.W # Width of full-connected layers
        self.Depth = self.net_cfg.network.nerf.D # depth of full-connected layers
        self.output_ch = output_ch # output channels alpha + rgb
        self.skips = skips # skip connections
        self.use_viewdirs = use_viewdirs

        self.cascade = self.net_cfg.task_arg.cascade_samples
        # fine network
        if len(self.cascade) > 1:
            self.fine_sample = self.cascade[1]
        else:
            self.fine_sample = None

        # 这里粗网络和细网络的参数结构都一样，只是传入的东西不太一样
        self.coarse_net = NeRF_network(output_ch = output_ch, skips = skips, use_viewdirs = use_viewdirs)
        self.fine_net = NeRF_network(output_ch = output_ch, skips = skips, use_viewdirs = use_viewdirs)
        if(len(self.cascade) > 1):
            self.fine_net = NeRF_network(output_ch = output_ch, skips = skips, use_viewdirs = use_viewdirs)
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
    
    def net_res(self, rays, viewdirs, fn):
        input = torch.reshape(rays, [-1, rays.shape[-1]])   # [N_rays * N_samples, 3]
        embedded = self.xyz_encoder(input)  # 对位置进行编码 [N_rays * N_samples, 63]
        if self.use_viewdirs:
            view_dirs = viewdirs[:, None].expand(rays.shape)
            view_dirs_flat = torch.reshape(view_dirs, [-1, view_dirs.shape[-1]])
            embedded_dirs = self.dir_encoder(view_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        output = fn(embedded)   # [65536. 4]
        output = torch.reshape(output, list(rays.shape[:-1]) + [output.shape[-1]])    # [N_rays, N_samples, 4]
        return output 
    pass

    def get_output(self, prev_res, z_vals, rays_d, white_bkgd = False):
        '''
        prev_res: the result from the network, [N_rays, N_samples, 4]
        z_vals: random sample between near to far
        rays_d: view directions
        '''
        dists = z_vals[...,1:] - z_vals[...,:-1]    # the distance between adjacent samples
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(device)], -1)    # (N_rays, N_samples)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        rgb = torch.sigmoid(prev_res[...,:3])       # rgb values
        sigma = prev_res[...,-1]    # \sigma in the formula
        getalpha = lambda raw, dists: 1.-torch.exp(-F.relu(raw) * dists)

        alpha = getalpha(raw=prev_res[:, :, 3], dists=dists)
        weights = alpha * torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]    # 增加的那行ones是为了让结果不包含alpha_n

        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]
        acc_map = torch.sum(weights, dim=-1)        # [1024] 权重和，用于计算白背景下的rgb
        
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])
        return rgb_map, acc_map, weights

    def hier_samp(self, mids, weights, N_samples):
        weights = weights + 1e-5    # prevent nans
        pdf = weights / torch.sum(weights, dim = -1, keepdim=True)  # keepdim是为了保证求和后的shape是[N_rays, 62]
        # accumulate sum
        cdf = torch.cumsum(pdf, dim = -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim = -1) # [N_rays, 63] 在最前面补一个0

        # uniform samples
        u_samples = torch.rand([cdf.shape[0], N_samples]).contiguous().to(device)
        
        # find the place in cdf
        # 分层采样的意义就在于，由于不同地方的权重不一样，在权重较大的区间，cdf累加和变化就大，u_samples落在对应区间的概率就比较大
        indexes = torch.searchsorted(cdf, u_samples, right=True)    # [N_rays, 128] right 是为了选择左闭右开区间
        lower = torch.max(indexes - 1, torch.zeros_like(indexes)).to(device)   # 防止出现0以下的值
        upper = torch.min(indexes, torch.ones_like(indexes) * (indexes.shape[-1] - 1)).to(device) # 防止超出边缘值, 在这里是<=62
        index_int = torch.stack([lower, upper], dim = -1)   # [N_rays, N_samples, 2]

        match_shape = [index_int.shape[0], index_int.shape[1], cdf.shape[-1]]
        # [N_rays, N_samples, 2]
        cdf_gather = torch.gather(cdf.unsqueeze(1).expand(match_shape), dim=2, index=index_int)
        mids_gather = torch.gather(mids.unsqueeze(1).expand(match_shape), dim=2, index=index_int)

        denom = cdf_gather[..., 1] - cdf_gather[..., 0]     # get the difference
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)    # 防止出现nans
        t = (u_samples - cdf_gather[..., 0]) / denom    # 线性差值，用于之后的o+td 这里是得到频率

        # [N_rays, N_samples]
        samples = mids_gather[..., 0] + t * (mids_gather[..., 1] - mids_gather[..., 0])
        return samples

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

        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        if lindisp:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)  # 线性采样
        else:   # 差值采样  
            z_vals = near * (1. - t_vals) + far * t_vals
        z_vals = z_vals.expand([N_rays, N_samples])
        if perturb > 0.:    # 增添一些随机性   
            mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            lower = torch.cat([z_vals[:, :1], mids], -1)
            upper = torch.cat([mids, z_vals[:, -1: ]], -1)

            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand
        
        # o + td
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., : , None]   # [N_rays, N_samples, 3]

        coarse_res = self.net_res(pts, viewdirs, self.coarse_net) # get the result from the coarse net [N_rays, N_samples, 4]
        
        rgb_map, acc_map, weights = self.get_output(coarse_res, z_vals, rays_d)

        if len(self.cascade) > 1:        # hierachical sampling needed
            mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])   # [N_rays, N_samples - 1]

            rgb_map0, acc_map0 = rgb_map, acc_map
            new_samples = self.hier_samp(mids, weights[:, 1:-1], self.cascade[1])   # [N_rays, N_samples]
            new_samples = new_samples.detach()
            z_vals = torch.sort(torch.cat([z_vals, new_samples], dim=-1), dim=-1)[0]    # sort and take the values, not the indices
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]    # p = o + td [N_rays, 64+128, 3]

            '''
            here we run the fine net
            '''
            fine_res = self.net_res(pts, viewdirs, self.fine_net)
            rgb_map, acc_map, weights = self.get_output(fine_res, z_vals, rays_d)
        
        ret = {'rgb_map': rgb_map, 'acc_map': acc_map}
        if len(self.cascade) > 1:
            ret['rgb0'] = rgb_map0
            ret['acc_map'] = acc_map0
            ret['z_std'] = torch.std(new_samples, dim = -1, unbiased = False)

        # for k in ret:
        #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
        #         print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def batchify(self, rays):   # 控制光线数量  分批处理
        chunk_size = self.net_cfg.task_arg.chunk_size
        N_samples = self.cascade[0] # N_samples = 64
        all_ret = {}
        for i in range(0, rays.shape[0], chunk_size):
            ret = self.render_rays(rays[i:i+chunk_size], N_samples, perturb=0.5)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        
        ret_val = {k: torch.cat(all_ret[k], 0) for k in all_ret}        # 把结果从列表的形式转成tensor
        return ret_val
        # pass
    def forward(self, batch, near = 2.0, far = 6.0):
        use_viewdirs = self.use_viewdirs
        rays_o, rays_d = batch['rays_o'], batch['rays_d']

        if use_viewdirs:
            viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)    # 规范化
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()    # [N_rays, 3]

        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near = near * torch.ones(rays_d.shape[0], device=rays_d.device)
        far = far * torch.ones(rays_d.shape[0], device=rays_d.device)

        rays = torch.cat([rays_o, rays_d, near[:, None], far[:, None]], dim=-1)    # [N_rays, 8]
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)    # [N_rays, 11] 如果要加上视角的话
        
        ret_val = self.batchify(rays)
        return ret_val