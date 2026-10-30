from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

import matplotlib.cm as cmx
import numpy as np
from mmdet3d.ops import bev_pool

NUM_COLORS = 30

def getcolors(number=NUM_COLORS):
    colorlist = list(cmx.rainbow(np.linspace(0, 1, number)))
    colorlist = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in colorlist]
    return colorlist

def draw_2d_points_to_image(show_file, image, draw_points_2d, depth):
    import cv2
    h, w, _ = image.shape
    pts = np.array(draw_points_2d)
    depth = np.array(depth)
    # print('pts', pts.shape, 'depth', depth.shape)
    # generate color map from depth
    colorlist = getcolors(NUM_COLORS)
    max_x = np.max(depth[:, 0])
    min_x = np.min(depth[:, 0])
    step_x = (max_x - min_x) / NUM_COLORS
    # draw points
    n = pts.shape[0]
    show_image = image.copy()
    count = 0
    for i in range(n):
        c = int((depth[i] - min_x) / step_x)
        c = c if c <= NUM_COLORS - 1 else NUM_COLORS - 1
        if (0 < int(pts[i][0]) < w) and (0 < int(pts[i][1]) < h):
            cv2.circle(show_image, (int(pts[i][0]), int(pts[i][1])), radius=2, color=colorlist[c], thickness=-1)
            count += 1
    cv2.imwrite(show_file, show_image)


__all__ = ["BaseTransform", "BaseDepthTransform"]


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [round((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx # 分辨率，左边界，gridsize


class BaseTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound#[1.0, 90.0, 0.5]

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels#80
        self.frustum = self.create_frustum()#[178, 32, 88, 3]
        self.D = self.frustum.shape[0]#(90-1.0)/0.5 = 178
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self):#生成视锥
        iH, iW = self.image_size#[256, 704]
        fH, fW = self.feature_size#[32, 88]

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )#[178, 32, 88]
        D, _, _ = ds.shape#(90-1.0)/0.5 = 178

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)#[178, 32, 88, 3]
        return nn.Parameter(frustum, requires_grad=False)

    @force_fp32()
    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation, 减掉图像增强后处理
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        ) # torch.Size([1, 1, 178, 32, 88, 3, 1])
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )

        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):#池化操作
        B, N, D, H, W, C = x.shape#[4, 1, 178, 32, 88, 80]
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        # self.nx: tensor([328, 280,   1]
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])# torch.Size([4, 80, 1, 328, 280])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        **kwargs,
    ):
        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )
        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def forward(#从这里开始看
        self,#lidar点云
        img,#camera特征
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        rots = sensor2ego[..., :3, :3]#对后两个维度进行切片，只保留三个（sensor2ego总共是多少维度的？）传感器到自车的旋转矩阵
        trans = sensor2ego[..., :3, 3]#平移矩阵
        intrins = cam_intrinsic[..., :3, :3]#camera内参
        post_rots = img_aug_matrix[..., :3, :3]#图像增强矩阵
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]#lidar到ego坐标转换
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]#camera到lidar坐标转换
        camera2lidar_trans = camera2lidar[..., :3, 3]

        # print('BaseDepthTransform forward ', len(points), img.shape, self.image_size, self.feature_size)# 4 torch.Size([4, 1, 256, 32, 88]) [256, 704] [32, 88]

        batch_size = len(points)#camera pipeline,这里把点云按照camera的内外参数投影过去，获取像素深度,其实没啥用
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(
            points[0].device
        )#创建一个(batch_size, img.shape[1], 1, *self.image_size)的全0张量 torch.Size([4, 1, 1, 256, 704])

        for b in range(batch_size):#batch_size
            cur_coords = points[b][:, :3].clone() #点云只要xyz前三维度，不要第四维度的intensity # torch.Size([n, 3])
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug, lidar数据增强矩阵的逆变换
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image, lidar坐标系转换到camera坐标系，再经过内参转换，转换到像素坐标系
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1) # torch.Size([1, 3, n])
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # # import ipdb; ipdb.set_trace()
            # ### project pcd to image
            # import cv2
            # import copy
            # for visidx in range(len(metas[b]['filename'])):
            #     name = metas[b]['timestamp']
            #     cur_coords_vis = copy.deepcopy(cur_coords)
            #     im_vis = cv2.imread(metas[b]['filename'][visidx])
            #     # print(metas[b]['filename'][visidx])
            #     draw_points_2d_vis = cur_coords_vis[visidx, :2, :].transpose(1, 0).cpu().numpy()
            #     depth_vis = cur_coords_vis[:, 2:3, :][visidx].transpose(1, 0).cpu().numpy()

            #     draw_points_2d_vis[np.isnan(draw_points_2d_vis)] = 0
            #     del_xindx, del_yindx = np.where(draw_points_2d_vis[:, :] == 0)
            #     draw_points_2d_vis = np.delete(draw_points_2d_vis, del_xindx, axis=0)
            #     depth_vis = np.delete(depth_vis, del_xindx, axis=0)
            #     draw_2d_points_to_image('vtrans_vis/%s_%d.jpg'%(name, visidx), im_vis, draw_points_2d_vis, depth_vis)
            # #####

            # imgaug， 按照image_aug进行变换
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2) # torch.Size([1, n, 2])

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]] # [v, u] 对应图片的 [h, w] # torch.Size([1, n, 2])

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)#uv分别于HW进行比较，判断是否在图片上
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(#生成一系列的3D点
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img, depth)# torch.Size([4, 1, 178, 32, 88, 80])
        x = self.bev_pool(geom, x) # torch.Size([4, 80, 328, 280])
        return x
