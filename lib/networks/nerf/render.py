import torch
import torch.nn as nn

def get_embedder(L, i = 0):
    
    pass
def batchify_rays(rays, chunk = 1024 * 32, **kwargs):   # 分批处理光线
    pass

def render(H, W, K, chunk=1024*32, batch=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,   # c2w_staticcam: 用于可视化视角效果
                  **kwargs):
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