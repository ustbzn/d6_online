import torch
import numpy as np

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class TransFusionBBoxCoder(BaseBBoxCoder):
    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 score_threshold=None,
                 code_size=8,
                 ):
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size

    def encode(self, dst_boxes, num_rotcls):
        targets = torch.zeros([dst_boxes.shape[0], self.code_size]).to(dst_boxes.device)
        targets[:, 0] = (dst_boxes[:, 0] - self.pc_range[0]) / (self.out_size_factor * self.voxel_size[0])
        targets[:, 1] = (dst_boxes[:, 1] - self.pc_range[1]) / (self.out_size_factor * self.voxel_size[1])
        # targets[:, 2] = (dst_boxes[:, 2] - self.post_center_range[2]) / (self.post_center_range[5] - self.post_center_range[2])
        targets[:, 3] = dst_boxes[:, 3].log()
        targets[:, 4] = dst_boxes[:, 4].log()
        targets[:, 5] = dst_boxes[:, 5].log()
        targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5  # bottom center to gravity center

        angle_per_class = 2 * np.pi / num_rotcls
        heading_angle = dst_boxes[:, 6] + np.pi
        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        rotcls =  (shift_angle / angle_per_class).floor().long()
        rotres = shift_angle - (rotcls.float() * angle_per_class + angle_per_class / 2)
        rotres_norm = rotres / (angle_per_class / 2)

        targets[:, 6] = rotcls
        targets[:, 7] = rotres_norm
        
        if self.code_size == 10:
            targets[:, 8:10] = dst_boxes[:, 7:]
        return targets

    def decode(self, heatmap, rotcls, rotres, dim, center, height, vel, num_rotcls, filter=False):
        """Decode bboxes.
        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, num_cls, num_proposals].
            rot (torch.Tensor): Rotation with the shape of
                [B, 1, num_proposals].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 3, num_proposals].
            center (torch.Tensor): bev center of the boxes with the shape of
                [B, 2, num_proposals]. (in feature map metric)
            hieght (torch.Tensor): height of the boxes with the shape of
                [B, 2, num_proposals]. (in real world metric)
            vel (torch.Tensor): Velocity with the shape of [B, 2, num_proposals].
            filter: if False, return all box without checking score and center_range
        Returns:
            list[dict]: Decoded boxes.
        """
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        # change size to real world metric
        center[:, 0, :] = center[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        center[:, 1, :] = center[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        # center[:, 2, :] = center[:, 2, :] * (self.post_center_range[5] - self.post_center_range[2]) + self.post_center_range[2]
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        height = height - dim[:, 2:3, :] * 0.5  # gravity center to bottom center

        rot_cls = rotcls.argmax(dim=1).unsqueeze(dim=1)

        angle_per_class = 2 * np.pi / num_rotcls
        shift_angle = rotres * (angle_per_class / 2) + rot_cls * angle_per_class
        heading_angle = torch.where(shift_angle < 0, 2 * np.pi + shift_angle, shift_angle)
        rot = heading_angle - np.pi
        

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'bboxes': boxes3d, # torch.Size([200, 7])
                'scores': scores, # torch.Size([200])
                'labels': labels # torch.Size([200])
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heatmap.device)
                
            normal_preds = final_box_preds.clone()
            normal_preds[..., 2] += 0.5 * normal_preds[..., 5]
            mask = (normal_preds[..., :3]  >=
                    self.post_center_range[:3]).all(2)
            mask &= (normal_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)
            
            predictions_dicts = []
            for i in range(heatmap.shape[0]):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts