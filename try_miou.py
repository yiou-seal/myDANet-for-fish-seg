import os

from PIL import Image
import numpy as np
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results
import predict_asclass



'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
# 定义一个函数，函数名字为get_all_excel，需要传入一个目录
def get_all_pth(dir):
    file_list = []
    for root_dir, sub_dir, files in os.walk(r'' + dir):
        # 对文件列表中的每一个文件进行处理，如果文件名字是以‘xlxs’结尾就
        # 认定为是一个excel文件，当然这里还可以用其他手段判断，比如你的excel
        # 文件名中均包含‘res’，那么if条件可以改写为
        for file in files:
            # if file.endswith('.xlsx') and 'res' in file:
            if file.endswith('.pth'):
                # 此处因为要获取文件路径，比如要把D:/myExcel 和res.xlsx拼接为
                # D:/myExcel/res.xlsx，因此中间需要添加/。python提供了专门的
                # 方法
                file_name = os.path.join(root_dir, file)
                # 把拼接好的文件目录信息添加到列表中
                file_list.append(file_name)
    return file_list


def trymiou(modelfilepathes):
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 2
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 2
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    name_classes    = ["background","fish"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    dataset_path  = './DeepFish2/datasets'

    image_ids       = open(os.path.join(dataset_path, "ImageSets_include_nofish/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir          = os.path.join(dataset_path, "SegmentationClass/")
    raw_dir = os.path.join(dataset_path, "JPEGImages/")

    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    ######!!!!!!!!!!!用到的模型路径就是predict_asclass.py里的那个模型路径
    # predicttool= predict_asclass.Predictbymodel()

    # if miou_mode == 0 or miou_mode == 1:
    #     if not os.path.exists(pred_dir):
    #         os.makedirs(pred_dir)
    #
    #     print("Load model.")
    #     deeplab = DeeplabV3()
    #     print("Load model done.")
    #
    #     print("Get predict result.")
    #     for image_id in tqdm(image_ids):
    #         image_path  = os.path.join(dataset_path, "JPEGImages/" + image_id + ".jpg")
    #         image       = Image.open(image_path)
    #         image       = deeplab.get_miou_png(image)
    #         image.save(os.path.join(pred_dir, image_id + ".png"))
    #     print("Get predict result done.")


    modelfilepathes=get_all_pth(modelfilepathes)
    if miou_mode == 0 or miou_mode == 2:
        maxiou=0
        maxpath=''
        for path in modelfilepathes:
            predicttool = predict_asclass.Predictbymodel(path)
            print("Get miou."+path)
            hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir,raw_dir, image_ids,image_ids, num_classes, name_classes,predicttool)  # 执行计算mIoU的函数
            curiou=np.nanmean(IoUs)
            if curiou>maxiou:
                maxiou=curiou
                maxpath=path

        predicttool = predict_asclass.Predictbymodel(maxpath)
        print("Get miou." + maxpath)
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, raw_dir, image_ids, image_ids, num_classes,
                                                        name_classes, predicttool)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

if __name__ == "__main__":
    modelfilepathes = '/home/xx316/zth/deeplabv3-plus-pytorch-main/imagenetpertrained_shujzengyi_excludenofish_lr5e_4_diceloss_logs'
    trymiou(modelfilepathes)