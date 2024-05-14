# from lzx
import cv2
import copy
import numpy as np
from paddle.vision.transforms import functional as F
from PIL import Image
import sys


def cv_imshow(name, img):
    """
    opencv辅助图片显示函数
    :param name: 显示图片名称
    :param img: 图片对象
    """
    size = img.shape
    w = int(size[1] / 2)
    h = int(size[0] / 2)
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name, img)
    cv2.imwrite("gray.bmp", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_picture_effect(picture_path, binary_thresh=130, binary_max=255, kernel_size=3, erode_batch=1):
    """
    获取图片有效部分，有效部分获取逻辑如下：
    将原图转换为灰度图进行二值处理，对二值处理后图片进行腐蚀操作，减轻图片中噪音对绘制轮廓的影响
    将进行腐蚀后的图片进行轮廓绘制，取最大轮廓绘制最小外接矩形
    根据该最小外接矩形的左顶点以及宽高对图片进行有效部分提取
    :param binary_thresh:二值操作阈值
    :param binary_max:二值操作最大值
    :param kernel_size:进行腐蚀操作的卷积核大小
    :param erode_batch:腐蚀操作次数
    :return:返回PIL.Image格式的有效图片数据
    """
    try:
        # 设置有效图片初始值
        effect_pic = None
        # 获取图片
        img = cv2.imread(picture_path)
        # 复制图片进行后续操作
        img2 = copy.deepcopy(img)

        # 对图像进行灰度图转换
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # TODO: 二值变换图片不明显
        # 对图片进行二值转换
        ret, binary = cv2.threshold(gray, binary_thresh, binary_max, cv2.THRESH_BINARY_INV)
        cv_imshow("erzhi", binary)

        # 进行腐蚀操作
        # 设置一个卷积核
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # 进行腐蚀操作
        erosion = cv2.erode(binary, kernel, iterations=erode_batch)
        cv_imshow('fushi', erosion)

        # 获取轮廓
        contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # 判断是否存在轮廓
        if contours:
            # 获取最大轮廓的索引
            area = map(cv2.contourArea, contours)
            area_list = list(area)
            area_max = max(area_list)
            post = area_list.index(area_max)
            # 获取最大索引对应轮廓
            cnt = contours[post]

            # 绘制外接矩形
            left_top_x, left_top_y, bound_w, bound_h = cv2.boundingRect(cnt)
            # 按外接矩形的大小进行裁剪
            effect_pic = F.crop(img, left_top_y, left_top_x, bound_h, bound_w)

            # # 转换图片格式为Image并返回
            # effect_pic = Image.fromarray(cv2.cvtColor(effect_pic, cv2.COLOR_BGR2RGB))
            return effect_pic
        else:
            return effect_pic

    except Exception as error:
        print('Error in get_picture_effect, error is {}'.format(error))
        sys.exit(0)

if __name__ == '__main__':
    effect_pic = get_picture_effect(r"picture_path")
    cv_imshow("effect", effect_pic)
