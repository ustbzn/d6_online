import mmcv
import torch
import numpy as np


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # logits2label, grid2point
        if "seg_logits_bev" in result[0]:
            # pred_labels_bev = torch.argmax(result[0]["seg_logits_bev"], dim=0)
            # pred_labels_bev = result[0]["seg_logits_bev"]
            # result[0]["pred_labels_bev"] = pred_labels_bev
            # # # output visabel seg result
            lidar_path = data["metas"].data[0][0]["lidar_path"]
            cloud_name = lidar_path.split('/')[-1][:-4]
            points = data["points"].data[0][0].detach().numpy()
            # points_grid_ind = data["points_grid_ind"].data[0][0].detach().numpy()
            points_labels = result[0]['pred_points_labels']
            # points_labels = pred_labels_bev[points_grid_ind[:, 0], points_grid_ind[:, 1]].detach().numpy()
            points_labeled = np.hstack((points, points_labels.reshape(-1,1)))
            # np.savetxt("/rbs/baozhiping/{}.txt".format(cloud_name), points_labeled, fmt='%.3f')
            # result[0]["pred_points_labels"] = points_labels
            # result[0]["gt_points_labels"] = data["points_labels"]

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
