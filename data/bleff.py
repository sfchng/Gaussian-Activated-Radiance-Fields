import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle

from . import base
import camera
from util import log,debug

class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        self.raw_H,self.raw_W = 1040,1560
        super().__init__(opt,split)
        self.root = opt.data.root or "data/bleff"
        self.path = "{}/{}/{}".format(self.root,opt.data.mode,opt.data.scene)
        
        """ BLEFF has gt_metas.json, containing training and testing """
        ## load/parse metadata ##
        meta_fname = "{}/gt_metas.json".format(self.path)
        with open(meta_fname) as file:
            self.meta = json.load(file)
        
        ## read ids ##
        if split != "train":
            ## TODO: ugly hack ##
            split = "val"
        img_ids = np.loadtxt("{}/{}_ids.txt".format(self.path, split)).astype(int)
        self.path_image = "{}/images".format(self.path)
        image_fnames = sorted(os.listdir(self.path_image))
        image_fnames = list(np.array(image_fnames)[img_ids])
        self.poses_raw = np.array(self.meta['c2ws']).astype(np.float32)  #(N,4,4)
        
        self.poses = self.parse_cameras_and_bounds(opt)
        self.poses = self.poses[img_ids]
        
        self.list = list(zip(image_fnames, self.poses))
        ## focal x ##
        self.focal_x = 0.5 * self.raw_W / np.tan(0.5*self.meta["cam_angle_x"])
        self.focal_y = 0.5 * self.raw_H / np.tan(0.5*self.meta["cam_angle_y"])        
        
        if opt.data.preload:
            self.images = self.preload_threading(opt, self.get_image)
            self.cameras = self.preload_threading(opt, self.get_camera, data_str="cameras")

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def parse_cameras_and_bounds(self,opt):
        # parse cameras (intrinsics and poses)
        poses_raw = self.poses_raw[:,:3]
        # roughly center camera poses
        poses_raw = self.center_camera_poses(opt,poses_raw)
        return poses_raw

    def center_camera_poses(self,opt,poses):
        # compute average pose
        poses = torch.tensor(poses, dtype=torch.float32)
        center = poses[...,3].mean(dim=0)
        v1 = torch_F.normalize(poses[...,1].mean(dim=0),dim=0)
        v2 = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
        v0 = v1.cross(v2)
        pose_avg = torch.stack([v0,v1,v2,center],dim=-1)[None] # [1,3,4]
        # apply inverse of averaged pose
        poses = camera.pose.compose([poses,camera.pose.invert(pose_avg)])
        return poses

    def get_all_camera_poses(self,opt):
        pose_raw_all = [tup[1] for tup in self.list]
        pose_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
        return pose_all

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        image = self.preprocess_image(opt,image,aug=aug)
        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx)
        intr,pose = self.preprocess_camera(opt,intr,pose,aug=aug)

        sample.update(
            image=image,
            intr=intr,
            pose=pose,
        )
        return sample

    def get_image(self,opt,idx):
        image_fname = "{}/{}".format(self.path_image,self.list[idx][0])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image

    def preprocess_image(self, opt, image, aug=None):
        """
        All blender datasets are stored four channels RGBA
        """
        image = super().preprocess_image(opt, image, aug)
        rgb, mask = image[:3], image[3:]
        if opt.data.bgcolor is not None:
            rgb = rgb*mask+opt.data.bgcolor*(1-mask)
        return rgb 

    def get_camera(self,opt,idx):
        intr = torch.tensor([[self.focal_x,0,self.raw_W/2],
                             [0,self.focal_y,self.raw_H/2],
                             [0,0,1]]).float()
        pose_raw = self.poses[idx]
        pose = self.parse_raw_camera(opt,pose_raw)
        return intr,pose

    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        pose = camera.pose.invert(pose)
        pose = camera.pose.compose([pose_flip,pose])
        return pose
