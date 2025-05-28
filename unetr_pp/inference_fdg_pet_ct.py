import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from scipy.ndimage import distance_transform_edt, binary_erosion


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def surface_distances(pred, gt):
    """
    计算两个二值图像之间的表面距离
    """
    pred = np.atleast_1d(pred.astype(np.bool_))
    gt = np.atleast_1d(gt.astype(np.bool_))

    if pred.shape != gt.shape:
        raise ValueError("预测和真实标签的形状必须相同")

    # 计算距离变换
    dist_pred = distance_transform_edt(pred)
    dist_gt = distance_transform_edt(gt)

    # 获取表面点
    pred_surface = pred & ~binary_erosion(pred)
    gt_surface = gt & ~binary_erosion(gt)

    # 计算表面距离
    dist_pred_to_gt = dist_pred[gt_surface]
    dist_gt_to_pred = dist_gt[pred_surface]

    return np.concatenate([dist_pred_to_gt, dist_gt_to_pred])


def hd95(pred, gt):
    """
    计算95% Hausdorff距离
    """
    if pred.sum() == 0 or gt.sum() == 0:
        return 0

    distances = surface_distances(pred, gt)
    if len(distances) == 0:
        return 0

    return np.percentile(distances, 95)


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        return hd95(pred, gt)
    else:
        return 0


def test(fold):
    path = '/mnt/disk0/smm/data/unetr_pp_raw/unetr_pp_raw_data/Task07_FDG_PET_CT'
    label_path = '/mnt/disk0/smm/data/unetr_pp_raw/unetr_pp_raw_data/Task07_FDG_PET_CT/labelsTs'
    pred_path = '/mnt/disk0/smm/data/unetr_pp_raw/unetr_pp_raw_data/Task07_FDG_PET_CT/infersTs'

    label_list = sorted(glob.glob(os.path.join(label_path, '*nii.gz')))
    infer_list = sorted(glob.glob(os.path.join(pred_path, '*nii.gz')))
    print("加载数据成功...")

    Dice_tumor = []
    HD_tumor = []

    result_dir = os.path.join(path, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fw = open(os.path.join(result_dir, 'fdp_pet_dice_five.txt'), 'w')

    for label_path, infer_path in zip(label_list, infer_list):
        case_name = infer_path.split('/')[-1]
        print(f"处理: {case_name}")

        label, infer = read_nii(label_path), read_nii(infer_path)

        # 肿瘤分割 (二分类, 标签1为肿瘤)
        label_tumor = (label == 1)
        infer_tumor = (infer == 1)

        # 计算Dice和HD
        dice_score = dice(infer_tumor, label_tumor)
        hd_score = hd(infer_tumor, label_tumor)

        Dice_tumor.append(dice_score)
        HD_tumor.append(hd_score)

        fw.write('*' * 20 + '\n')
        fw.write(f"{case_name}\n")
        fw.write(f'肿瘤Dice: {dice_score:.4f}\n')
        fw.write(f'肿瘤HD95: {hd_score:.4f}\n')
        fw.write('*' * 20 + '\n')

    # 计算平均值
    avg_dice = np.mean(Dice_tumor)
    avg_hd = np.mean(HD_tumor)

    fw.write('\n总体评估:\n')
    fw.write(f'平均Dice: {avg_dice:.4f}\n')
    fw.write(f'平均HD95: {avg_hd:.4f}\n')

    print(f'平均Dice: {avg_dice:.4f}')
    print(f'平均HD95: {avg_hd:.4f}')

    fw.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("fold", help="fold name")
    args = parser.parse_args()
    fold = '0'
    test(fold)