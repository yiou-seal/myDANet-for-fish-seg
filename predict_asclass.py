#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from deeplab import DeeplabV3

class Predictbymodel:

    def __init__(self,modelpath='/home/xx316/zth/deeplabv3-plus-pytorch-main/fcn16_shujzengyi_excludenofish_lr1e_4_adam_logs/ep006-loss0.402-val_loss0.274.pth'):

        # -------------------------------------------------------------------------#
        #   如果想要修改对应种类的颜色，到generate函数里修改self.colors即可
        #   权值文件路径
        # -------------------------------------------------------------------------#
        self.deeplab = DeeplabV3(modelpath)
        # ----------------------------------------------------------------------------------------------------------#
        #   mode用于指定测试的模式：
        #   'predict'表示单张图片预测，如果想对预测过程进行修改
        #   ，如保存图片，截取对象等，可以先看下方详细的注释
        #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
        #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
        #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
        # ----------------------------------------------------------------------------------------------------------#
        self.mode = "predict"
        # ----------------------------------------------------------------------------------------------------------#
        #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
        #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
        #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
        #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
        #   video_fps用于保存的视频的fps
        #   video_path、video_save_path和video_fps仅在mode='video'时有效
        #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
        # ----------------------------------------------------------------------------------------------------------#
        self.video_path = 0
        self.video_save_path = ""
        self.video_fps = 25.0
        # -------------------------------------------------------------------------#
        #   test_interval用于指定测量fps的时候，图片检测的次数
        #   理论上test_interval越大，fps越准确。
        # -------------------------------------------------------------------------#
        self.test_interval = 100
        # -------------------------------------------------------------------------#
        #   dir_origin_path指定了用于检测的图片的文件夹路径
        #   dir_save_path指定了检测完图片的保存路径
        #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
        # -------------------------------------------------------------------------#
        self.dir_origin_path = "/home/xx316/zth/deeplabv3-plus-pytorch-main/DeepFish2/datasets/JPEGImages/"
        self.dir_redmask_path = "/home/xx316/zth/deeplabv3-plus-pytorch-main/DeepFish2/datasets/redSegmentationClass/"
        self.dir_mask_path = "/home/xx316/zth/deeplabv3-plus-pytorch-main/DeepFish2/datasets/redSegmentationClass/"
        self.dir_save_path = "/home/xx316/zth/deeplabv3-plus-pytorch-main/img_out/"
        self.dir_save_pathforcalIOU = "/home/xx316/zth/deeplabv3-plus-pytorch-main/img_out/"
        # -------------------------------------------------------------------------#
        #   以下指定了从何处读取val.txt
        # -------------------------------------------------------------------------#

        # -------------------------------------------------------------------------#
        #   以下指定了生成什么
        # -------------------------------------------------------------------------#

    def predict_image_list(self,imglist,raws=False,rescale=2):
        '''
        predict.py有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''

        for img in imglist:
        # while True:
            # img = input('Input image filename:')

            try:

                image = Image.open(self.dir_origin_path + img + ".jpg")
                image.save(self.dir_save_path + img + ".jpg")
                mask=Image.open(self.dir_redmask_path + img + '.png')
                mask.save(self.dir_save_path + 'mask_' + img + ".jpg")

            except:
                print('Open Error! Try again!')
                continue
            else:
                if raws:
                    r_image = self.deeplab.detect_image_inrawsize(image)
                else:
                    r_image,maskimage = self.deeplab.detect_image_returntwo(image,rescale)

                # r_image.show()
                r_image.save(self.dir_save_path+'predic_'+img+".jpg")
                maskimage.save(self.dir_save_path + 'predic_mask_' + img + ".jpg")

    def predict_one_image(self,image,rescale=2):
        '''
        predict.py有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''


        # try:

            # image = Image.open(self.dir_origin_path + img + ".jpg")
            # image.save(self.dir_save_path + img + ".jpg")
            # mask=Image.open(self.dir_redmask_path + img + '.png')
            # mask.save(self.dir_save_path + 'mask_' + img + ".jpg")

        # except:
        #     print('Open Error! Try again!')
        #     return None
        # else:
        r_image = self.deeplab.detect_image_returnndarray(image,rescale)
        return r_image
        # r_image.show()
        # r_image.save(self.dir_save_path+'predic_'+img+".jpg")


if __name__ == "__main__":
    imglist = ['9908_Acanthopagrus_palmaris_f000040',
               '9862_no_fish_f000170',
               '9866_acanthopagrus_and_caranx_f000160','7398_F6_f000070','7482_F3_f000080','7482_F2_f000560']
    # imglist = ['7482_F2_f000560']
    # imglist = ['7482_F2_f000560'] # 两个注意力模块用的那张显示效果的图
    # imglist=['7482_F2_f000560']
    # imglist=['7398_F6_f000070']
    preder=Predictbymodel()
    preder.predict_image_list(imglist)