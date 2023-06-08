"""
This model predicts spherical harmonics coefficients, which will then be evaluated to obtain the corresponding rgb colour
"""
import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict

import lpips
from external.pytorch_ssim import pytorch_ssim

import util,util_vis
from util import log,debug
from . import base
import camera

import sh


# ============================ main engine for training and evaluation ============================

class Model(base.Model):

    def __init__(self,opt):
        super().__init__(opt)
        self.lpips_loss = lpips.LPIPS(net="alex").to(opt.device)

    def load_dataset(self,opt,eval_split="val"):
        super().load_dataset(opt,eval_split=eval_split)
        # prefetch all training data
        self.train_data.prefetch_all_data(opt)
        self.train_data.all = edict(util.move_to_device(self.train_data.all,opt.device))

    def setup_optimizer(self,opt):
        log.warning("setting up optimizer {}...".format(opt.optim.algo))
        optimizer = getattr(torch.optim,opt.optim.algo)
        self.optim = optimizer([dict(params=self.graph.nerf.parameters(),lr=opt.optim.lr)])
        if opt.nerf.fine_sampling:
            self.optim.add_param_group(dict(params=self.graph.nerf_fine.parameters(),lr=opt.optim.lr))
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched.type)
            if opt.optim.lr_end:
                assert(opt.optim.sched.type=="ExponentialLR")
                opt.optim.sched.gamma = (opt.optim.lr_end/opt.optim.lr)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim,**kwargs)

    def train(self,opt):
        # before training
        log.title("TRAINING for GARF-SH START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.graph.train()
        self.ep = 0 # dummy for timer
        
        # training
        if self.iter_start==0: self.validate(opt,0)
        loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
        for self.it in loader:
            if self.it<self.iter_start: continue
            # set var to all available images
            var = self.train_data.all
            self.train_iteration(opt,var,loader)
            ## linearly adjust the learning rate if scheduler is not set ##
            if opt.optim.sched: 
                self.sched.step()
            else:
                decay_steps = opt.optim.lr_decay * 1000
                new_lrate = opt.optim.lr * (0.1 ** (self.it/decay_steps))
                self.optim.param_groups[0]['lr'] = new_lrate
            if self.it%opt.freq.val==0: 
                self.validate(opt,self.it)
                self.validate_training_state(opt, self.it)
            if self.it%opt.freq.ckpt==0: self.save_checkpoint(opt,ep=None,it=self.it)
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        # log learning rate
        if split=="train":
            lr = self.optim.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr"),lr,step)
            if opt.nerf.fine_sampling:
                lr = self.optim.param_groups[1]["lr"]
                self.tb.add_scalar("{0}/{1}".format(split,"lr_fine"),lr,step)
        # compute PSNR
        if opt.loss.psnr:
            psnr = -loss.render
        else:
            psnr = -10*loss.render.log10()
        self.tb.add_scalar("{0}/{1}".format(split,"PSNR"),psnr,step)
        if opt.nerf.fine_sampling:
            psnr = -10*loss.render_fine.log10()
            self.tb.add_scalar("{0}/{1}".format(split,"PSNR_fine"),psnr,step)

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train",eps=1e-10):
        if opt.tb:
            if not opt.nerf.rand_rays or split!="train":
                util_vis.tb_image(opt,self.tb,step,split,"image",var.image)
                invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
                rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                util_vis.tb_image(opt,self.tb,step,split,"rgb",rgb_map)
                util_vis.tb_image(opt,self.tb,step,split,"invdepth",invdepth_map)
                if opt.nerf.fine_sampling:
                    invdepth = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
                    rgb_map = var.rgb_fine.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                    invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                    util_vis.tb_image(opt,self.tb,step,split,"rgb_fine",rgb_map)
                    util_vis.tb_image(opt,self.tb,step,split,"invdepth_fine",invdepth_map)


    @torch.no_grad()
    def visualize_train(self,opt,var,step=0,split="train",eps=1e-10):
        if opt.tb:
            util_vis.tb_image(opt,self.tb,step,split,"image",var.image)
            invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
            rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
            util_vis.tb_image(opt,self.tb,step,split,"rgb",rgb_map)
            util_vis.tb_image(opt,self.tb,step,split,"invdepth",invdepth_map)
            if opt.nerf.fine_sampling:
                invdepth = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
                rgb_map = var.rgb_fine.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                util_vis.tb_image(opt,self.tb,step,split,"rgb_fine",rgb_map)
                util_vis.tb_image(opt,self.tb,step,split,"invdepth_fine",invdepth_map)

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        return None,pose_GT

    @torch.no_grad()
    def evaluate_full(self,opt,eps=1e-10):
        self.graph.eval()
        loader = tqdm.tqdm(self.test_loader,desc="evaluating",leave=False)
        res = []
        test_path = "{}/test_view".format(opt.output_path)
        os.makedirs(test_path,exist_ok=True)
        for i,batch in enumerate(loader):
            var = edict(batch)
            var = util.move_to_device(var,opt.device)
            if opt.model=="garf_sh" and opt.optim.test_photo:
                # run test-time optimization to factorize imperfection in optimized poses from view synthesis evaluation
                var = self.evaluate_test_time_photometric_optim(opt,var)
            var = self.graph.forward(opt,var,mode="eval")
            # evaluate view synthesis
            invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
            rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
            psnr = -10*self.graph.MSE_loss(rgb_map,var.image).log10().item()
            ssim = pytorch_ssim.ssim(rgb_map,var.image).item()
            lpips = self.lpips_loss(rgb_map*2-1,var.image*2-1).item()
            res.append(edict(psnr=psnr,ssim=ssim,lpips=lpips))
            # dump novel views
            torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(test_path,i))
            torchvision_F.to_pil_image(var.image.cpu()[0]).save("{}/rgb_GT_{}.png".format(test_path,i))
            torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save("{}/depth_{}.png".format(test_path,i))
        # show results in terminal
        log.info("evaluate test set...")
        print("----------------------------------------------------------")
        message = util.green("PSNR:", bold=True)  
        message += "{:8.2f}\n".format(np.mean([r.psnr for r in res]))    
        message += util.green("SSIM:", bold=True)  
        message += "{:8.2f}\n".format(np.mean([r.ssim for r in res]))   
        message += util.green("LPIPS:", bold=True)  
        message += "{:8.2f}".format(np.mean([r.lpips for r in res]))   
        print(message) 
        print("----------------------------------------------------------")
        # dump numbers to file
        quant_fname = "{}/quant.txt".format(opt.output_path)
        with open(quant_fname,"w") as file:
            for i,r in enumerate(res):
                file.write("{} {} {} {}\n".format(i,r.psnr,r.ssim,r.lpips))


    @torch.no_grad()
    def generate_videos_synthesis(self,opt,eps=1e-10):
        self.graph.eval()
        if opt.data.dataset=="blender":
            test_path = "{}/test_view".format(opt.output_path)
            # assume the test view synthesis are already generated
            print("writing videos...")
            rgb_vid_fname = "{}/test_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/test_view_depth.mp4".format(opt.output_path)
            os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,rgb_vid_fname))
            os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,depth_vid_fname))
        else:
            pose_pred,pose_GT = self.get_all_training_poses(opt)
            poses = pose_pred if opt.model=="garf_sh" else pose_GT
            if opt.model=="garf_sh" and opt.data.dataset=="llff":
                _,sim3 = self.prealign_cameras(opt,pose_pred,pose_GT)
                scale = sim3.s1/sim3.s0
            else: scale = 1
            # rotate novel views around the "center" camera of all poses
            idx_center = (poses-poses.mean(dim=0,keepdim=True))[...,3].norm(dim=-1).argmin()
            pose_novel = camera.get_novel_view_poses(opt,poses[idx_center],N=60,scale=scale).to(opt.device)
            # render the novel views
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path,exist_ok=True)
            pose_novel_tqdm = tqdm.tqdm(pose_novel,desc="rendering novel views",leave=False)
            intr = edict(next(iter(self.test_loader))).intr[:1].to(opt.device) # grab intrinsics
            for i,pose in enumerate(pose_novel_tqdm):
                ret = self.graph.render_by_slices(opt,pose[None],intr=intr) if opt.nerf.rand_rays else \
                      self.graph.render(opt,pose[None],intr=intr)
                invdepth = (1-ret.depth)/ret.opacity if opt.camera.ndc else 1/(ret.depth/ret.opacity+eps)
                rgb_map = ret.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(novel_path,i))
                torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save("{}/depth_{}.png".format(novel_path,i))
            # write videos
            print("writing videos...")
            rgb_vid_fname = "{}/novel_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/novel_view_depth.mp4".format(opt.output_path)
            os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,rgb_vid_fname))
            os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,depth_vid_fname))

# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.nerf = NeRF_Gaussian_SH(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF_Gaussian_SH(opt)

    def forward(self,opt,var,mode=None):
        batch_size = len(var.idx)
        pose = self.get_pose(opt,var,mode=mode)
        ## render images ##
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            ## sample random rays for optimization ##
            var.ray_idx = torch.randperm(opt.H*opt.W,device=opt.device)[:opt.nerf.rand_rays//batch_size]
            ret = self.render(opt,pose,intr=var.intr,ray_idx=var.ray_idx,mode=mode) # [B,N,3],[B,N,1]
        else:
            ## render full image (process in slices) ##
            ret = self.render_by_slices(opt,pose,intr=var.intr,mode=mode) if opt.nerf.rand_rays else \
                  self.render(opt,pose,intr=var.intr,mode=mode) # [B,HW,3],[B,HW,1]
        var.update(ret)
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image.view(batch_size,3,opt.H*opt.W).permute(0,2,1)
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            image = image[:,var.ray_idx]
        # compute image losses
        if opt.loss_weight.render is not None:
            if opt.loss.psnr:
                loss.render = self.PSNR_loss(var.rgb,image)
            else:
                loss.render = self.MSE_loss(var.rgb,image)
        if opt.loss_weight.render_fine is not None:
            assert(opt.nerf.fine_sampling)
            loss.render_fine = self.MSE_loss(var.rgb_fine,image)
        return loss

    def get_pose(self,opt,var,mode=None):
        return var.pose

    def render(self,opt,pose,intr=None,ray_idx=None,mode=None):
        batch_size = len(pose)
        center,ray = camera.get_center_and_ray(opt,pose,intr=intr) # [B,HW,3]
        while ray.isnan().any(): # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
            center,ray = camera.get_center_and_ray(opt,pose,intr=intr) # [B,HW,3]
        if ray_idx is not None:
            # consider only subset of rays
            center,ray = center[:,ray_idx],ray[:,ray_idx]
        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
        # render with main MLP
        depth_samples = self.sample_depth(opt,batch_size,num_rays=ray.shape[1]) # [B,HW,N,1]
        rgb_samples,density_samples = self.nerf.forward_samples(opt,center,ray,depth_samples,mode=mode)
        rgb,depth,opacity,prob = self.nerf.composite(opt,ray,rgb_samples,density_samples,depth_samples)
        ret = edict(rgb=rgb,depth=depth,opacity=opacity) # [B,HW,K]
        # render with fine MLP from coarse MLP
        if opt.nerf.fine_sampling:
            with torch.no_grad():
                # resample depth acoording to coarse empirical distribution
                depth_samples_fine = self.sample_depth_from_pdf(opt,pdf=prob[...,0]) # [B,HW,Nf,1]
                depth_samples = torch.cat([depth_samples,depth_samples_fine],dim=2) # [B,HW,N+Nf,1]
                depth_samples = depth_samples.sort(dim=2).values
            rgb_samples,density_samples = self.nerf_fine.forward_samples(opt,center,ray,depth_samples,mode=mode)
            rgb_fine,depth_fine,opacity_fine,_ = self.nerf_fine.composite(opt,ray,rgb_samples,density_samples,depth_samples)
            ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [B,HW,K]
        return ret

    def render_by_slices(self,opt,pose,intr=None,mode=None):
        ret_all = edict(rgb=[],depth=[],opacity=[])
        if opt.nerf.fine_sampling:
            ret_all.update(rgb_fine=[],depth_fine=[],opacity_fine=[])
        # render the image by slices for memory considerations
        for c in range(0,opt.H*opt.W,opt.nerf.rand_rays):
            ray_idx = torch.arange(c,min(c+opt.nerf.rand_rays,opt.H*opt.W),device=opt.device)
            ret = self.render(opt,pose,intr=intr,ray_idx=ray_idx,mode=mode) # [B,R,3],[B,R,1]
            for k in ret: ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=1)
        return ret_all

    def sample_depth(self,opt,batch_size,num_rays=None):
        depth_min,depth_max = opt.nerf.depth.range
        num_rays = num_rays or opt.H*opt.W
        rand_samples = torch.rand(batch_size,num_rays,opt.nerf.sample_intvs,1,device=opt.device) if opt.nerf.sample_stratified else 0.5
        rand_samples += torch.arange(opt.nerf.sample_intvs,device=opt.device)[None,None,:,None].float() # [B,HW,N,1]
        depth_samples = rand_samples/opt.nerf.sample_intvs*(depth_max-depth_min)+depth_min # [B,HW,N,1]
        depth_samples = dict(
            metric=depth_samples,
            inverse=1/(depth_samples+1e-8),
        )[opt.nerf.depth.param]
        return depth_samples

    def sample_depth_from_pdf(self,opt,pdf):
        depth_min,depth_max = opt.nerf.depth.range
        # get CDF from PDF (along last dimension)
        cdf = pdf.cumsum(dim=-1) # [B,HW,N]
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]),cdf],dim=-1) # [B,HW,N+1]
        # take uniform samples
        grid = torch.linspace(0,1,opt.nerf.sample_intvs_fine+1,device=opt.device) # [Nf+1]
        unif = 0.5*(grid[:-1]+grid[1:]).repeat(*cdf.shape[:-1],1) # [B,HW,Nf]
        idx = torch.searchsorted(cdf,unif,right=True) # [B,HW,Nf] \in {1...N}
        # inverse transform sampling from CDF
        depth_bin = torch.linspace(depth_min,depth_max,opt.nerf.sample_intvs+1,device=opt.device) # [N+1]
        depth_bin = depth_bin.repeat(*cdf.shape[:-1],1) # [B,HW,N+1]
        depth_low = depth_bin.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        depth_high = depth_bin.gather(dim=2,index=idx.clamp(max=opt.nerf.sample_intvs)) # [B,HW,Nf]
        cdf_low = cdf.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        cdf_high = cdf.gather(dim=2,index=idx.clamp(max=opt.nerf.sample_intvs)) # [B,HW,Nf]
        # linear interpolation
        t = (unif-cdf_low)/(cdf_high-cdf_low+1e-8) # [B,HW,Nf]
        depth_samples = depth_low+t*(depth_high-depth_low) # [B,HW,Nf]
        return depth_samples[...,None] # [B,HW,Nf,1]

class NeRF_Gaussian_SH(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)


    def define_network(self,opt):

        input_3D_dim = 3
        output_dim = 3

        if opt.nerf.sh.deg >= 0:
            assert opt.nerf.sh.enabled == 1, ("You can only use up to one of: SH, oruse_viewdirs")

            output_dim *= (opt.nerf.sh.deg+1)**2

        self.gaussian_linear1 = torch.nn.Linear(input_3D_dim, opt.arch.width)

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(opt.arch.width, opt.arch.width)] + [torch.nn.Linear(opt.arch.width, opt.arch.width) if i not in opt.arch.skip else torch.nn.Linear(opt.arch.width + opt.arch.width, opt.arch.width) for i in range(opt.arch.depth-1)])

        self.output_linear = torch.nn.Linear(opt.arch.width, output_dim+1)

        if opt.init.weight.uniform:
            log.warning("initializing weights uniformly in range -0.1 to 0.1...")
            torch.nn.init.uniform_(self.gaussian_linear1.weight, -opt.init.weight.range,opt.init.weight.range)

            for i, layer in enumerate(self.pts_linears):
                torch.nn.init.uniform_(self.pts_linears[i].weight, -opt.init.weight.range,opt.init.weight.range)
    

    def forward(self,opt,points_3D,ray_unit=None,mode=None): # [B,...,3]

        feat = self.gaussian_init(opt,points_3D)
        points_enc = feat

        for i, l in enumerate(self.pts_linears):
            feat = self.pts_linears[i](feat)
            feat = self.gaussian(opt,feat)
            if i in opt.arch.skip:
                feat = torch.cat([points_enc, feat], -1)

     
        output = self.output_linear(feat)
        raw_density = output[...,0]
        raw_rgb = output[...,1:] #(B,rand_rays,N_samples, [(sh_deg+1)**2]*3)

        if opt.nerf.density_noise_reg and mode == "train":
            raw_density += torch.randn_like(raw_density)*opt.nerf.density_noise_reg 
        density_activ = getattr(torch_F, opt.arch.density_activ)   
        density = density_activ(raw_density) #(B,rand_rays,N_samples)    

        rgb = self.eval_points(opt,raw_rgb,ray_unit) #(B,rand_rays,N_samples,3)

        return rgb,density


    def forward_samples(self,opt,center,ray,depth_samples,mode=None):
        points_3D_samples = camera.get_3D_points_from_depth(opt,center,ray,depth_samples,multi_samples=True) # [B,HW,N,3]
        if opt.nerf.view_dep:
            ray_unit = torch_F.normalize(ray,dim=-1) # [B,HW,3]
            ray_unit_samples = ray_unit[...,None,:].expand_as(points_3D_samples) # [B,HW,N,3]
        else: ray_unit_samples = None
        rgb_samples,density_samples = self.forward(opt,points_3D_samples,ray_unit=ray_unit_samples,mode=mode) # [B,HW,N],[B,HW,N,3]
        return rgb_samples,density_samples

    def eval_points(self,opt,raw_rgb,viewdirs):
        """
        Compute view-dependent rgb color using hardcoded spherical harmonics polynomials.
        Args:
            opt (dict)
            raw_rgb (torch.Tensor[B,rand_rays,N_samples,[(sh_deg+1)**2]*3])
            viewdirs (torch.Tensor[B,rand_rays,N_samples,3)
        Return:
            rgb (torch.Tensor[B,rand_rays,N_samples,3)
        """
        if opt.nerf.sh.deg >= 0:
            assert opt.nerf.view_dep is not None

            raw_rgb = sh.eval_sh(opt.nerf.sh.deg, raw_rgb.reshape(*raw_rgb.shape[:-1],-1,(opt.nerf.sh.deg+1)**2),viewdirs )

        rgb = raw_rgb.sigmoid_()
        return rgb

    def composite(self,opt,ray,rgb_samples,density_samples,depth_samples):
        ray_length = ray.norm(dim=-1,keepdim=True) # [B,HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[...,1:,0]-depth_samples[...,:-1,0] # [B,HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples,torch.empty_like(depth_intv_samples[...,:1]).fill_(1e10)],dim=2) # [B,HW,N]
        dist_samples = depth_intv_samples*ray_length # [B,HW,N]
        sigma_delta = density_samples*dist_samples # [B,HW,N]
        alpha = 1-(-sigma_delta).exp_() # [B,HW,N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)).exp_() # [B,HW,N]
        prob = (T*alpha)[...,None] # [B,HW,N,1]
        # integrate RGB and depth weighted by probability
        depth = (depth_samples*prob).sum(dim=2) # [B,HW,1]
        rgb = (rgb_samples*prob).sum(dim=2) # [B,HW,3]
        opacity = prob.sum(dim=2) # [B,HW,1]
        if opt.nerf.setbg_opaque:
            rgb = rgb+opt.data.bgcolor*(1-opacity)
        return rgb,depth,opacity,prob # [B,HW,K]


    def gaussian_init(self,opt,x):
        x_ = self.gaussian_linear1(x)
        mu = torch.mean(x_, axis = -1).unsqueeze(-1)
        out = (-0.5*(x_-mu)**2/opt.arch.gaussian.sigma**2).exp()
        return out
    

    def gaussian(self,opt,x):
        """
        Args:
            opt
            x (torch.Tensor [B,num_rays,])
        """
        out = (-0.5*(x)**2/opt.arch.gaussian.sigma**2).exp()
        return out
