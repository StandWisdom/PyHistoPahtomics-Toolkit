import pandas as pd
import os,re,time
import cv2
import numpy as np
from tqdm import tqdm
import platform
from multiprocessing import Pool
import shutil
import matplotlib.pyplot as plt
from torchvision import transforms
import torchstain
import torch

import warnings
warnings.filterwarnings("ignore")

class PreOptWSI():
    def __init__(self):
        self.loadColorNormTag = 0 
        print('Primary_pro_WSI. Contain processing for H&E or IHC.')
    
    def seg_threshold(self,region):
        '''
        segmentation by digital image processing
        '''
        img=cv2.medianBlur(region,3) # 均值滤波
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_color = np.array([0, 0, 0])
        upper_color = np.array([255, 10, 255])
        mask0 = cv2.inRange(hsv, lower_color, upper_color)
        # plt.figure()
        # plt.imshow(mask0,cmap='binary')

        lower_color = np.array([0, 0, 0])
        upper_color = np.array([255, 40, 40])
        mask1 = cv2.inRange(hsv, lower_color, upper_color)
        # plt.figure()
        # plt.imshow(mask1,cmap='binary')

        lower_color = np.array([0, 0, 0])
        upper_color = np.array([255, 20, 60])
        mask2 = cv2.inRange(hsv, lower_color, upper_color)
        # plt.figure()
        # plt.imshow(mask2,cmap='binary')

        # 去除孔洞
        # lower_color = np.array([0, 0, 215])
        # upper_color = np.array([255, 40, 255])
        # mask3 = cv2.inRange(hsv, lower_color, upper_color)

        mask = cv2.bitwise_or(mask0, mask1, mask2)
        return ~mask

    def seg_threshold2(self, region):
        '''
        segmentation by digital image processing
        '''
        img = cv2.medianBlur(region, 9)  # 均值滤波
        mask = self.seg_color(img)
        return mask
    
    def seg_color(self,region):
        '''
        segmentation by color
        '''
        img=region
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # 黑
        lower_color = np.array([0, 0, 0])
        upper_color = np.array([255, 20, 155])
        mask0 = cv2.inRange(hsv, lower_color, upper_color)

        lower_color = np.array([0, 0, 0])
        upper_color = np.array([255, 255, 20])
        mask1 = cv2.inRange(hsv, lower_color, upper_color)

        # 红色
        # lower_color = np.array([125, 0, 0])
        # upper_color = np.array([185, 255, 255])
        # mask = cv2.inRange(hsv, lower_color, upper_color)

        mask = cv2.bitwise_or(mask0, mask1)
        return mask
    
    def seg_otsu(self,region):
        # region = cv2.medianBlur(region, 3)  # 均值滤波
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # 1:backgroun; 0:foreground
        return mask
    
    def FindMaxRegion(self,mask):        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 计算每个轮廓的面积并找到最大面积的轮廓
        max_area = 0
        max_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        
        # 创建一个全0的mask来绘制最大面积的轮廓
        max_area_mask = np.zeros_like(mask)
        
        # 将最大面积的轮廓填充为1
        cv2.drawContours(max_area_mask, [max_contour], -1, 1, -1)
        return max_area_mask

    def FindThresholdRegion(self, mask, thd):
        '''
        thd: rate of max contour area
        '''
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 这里的参数很重要
        # 计算每个轮廓的面积并找到最大面积的轮廓
        max_area = 0
        max_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # 创建一个全0的mask来绘制最大面积的轮廓
        max_area_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) > thd * max_area: # 0.05:
                # 将最大面积的轮廓填充为1
                cv2.drawContours(max_area_mask, [contour], -1, 1, -1)
        max_area_mask = cv2.bitwise_and(max_area_mask, max_area_mask, mask=mask)
        return max_area_mask
      
    def SuperimposedContour(self, img, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0,0,255), 2)
        return img   
    
    # ColorNorm
    def color_norm(self, img):
        # st = time.time()
        target = cv2.resize(cv2.cvtColor(cv2.imread("./data/target.png"), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))
        to_transform = img

        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255)
        ])
        
        torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        torch_normalizer.fit(T(target))        
        t_to_transform = T(to_transform)
        norm, H, E = torch_normalizer.normalize(I=t_to_transform, stains=True)
        # print(time.time()-st)
        return norm, H, E
    
    def resolution_norm(self,slide,coor,mpp_std,tilesize):    
        mpp = [slide.properties['openslide.mpp-x'],slide.properties['openslide.mpp-y']] # mpp
        mpp = np.around(np.array(mpp).astype(float),3)# mpp

        x_tilesize = np.around(tilesize/mpp_std[0]*mpp[0]).astype(np.uint)
        y_tilesize = np.around(tilesize/mpp_std[1]*mpp[1]).astype(np.uint)        
        slide_sub = slide.read_region((coor[0],coor[1]),0,(x_tilesize,y_tilesize))
        slide_sub = np.array(slide_sub,dtype=np.uint8)[:,:,:3]
        slide_sub = cv2.resize(slide_sub,(tilesize,tilesize))
        return np.array(slide_sub)
    
    def read_pathes_of_WSI(self, slide, coor, mpp_std, tilesize, mag=1):
        # parameters of wsi
        dim = list(slide.level_dimensions)
        dsample = list(slide.level_downsamples)  # 每一个级别K的对应的下采样因子，下采样因子应该对应一个倍率

        mpp = [slide.properties['openslide.mpp-x'],slide.properties['openslide.mpp-y']] # mpp
        mpp = np.around(np.array(mpp).astype(float),5)# mpp

        # 优化计算速度
        arr = np.abs(np.array(dsample)-mag)
        index = arr.argmin()
        [w, h] = dim[index]

        x_tilesize_rate = (mpp_std[0] * mag) / (mpp[0] * dsample[index]) # w
        y_tilesize_rate = (mpp_std[1] * mag) / (mpp[1] * dsample[index]) # h
        x_tilesize = np.around(tilesize * x_tilesize_rate).astype(np.uint32)
        y_tilesize = np.around(tilesize * y_tilesize_rate).astype(np.uint32)

        if mag == 1:
            loc_bias = 0
        else:
            # loc_bias = int(tilesize/dsample[index]*0.5)
            loc_bias = int((tilesize/2)*(mag-1))

        x_wsi = int(coor[0] - loc_bias)
        y_wsi = int(coor[1] - loc_bias)

        try:
            '''
            如果错误大概率低倍率cut时坐标越界了
            '''
            # if x_wsi + x_tilesize > w:
            #     slide_sub = slide.read_region((w - x_tilesize, y_wsi), index, (x_tilesize, y_tilesize))
            # elif y_wsi + y_tilesize > h:
            #     slide_sub = slide.read_region((x_wsi, h - y_tilesize), index, (x_tilesize, y_tilesize))
            # elif x_wsi + x_tilesize > w and y_wsi + y_tilesize > h:
            #     slide_sub = slide.read_region((w - x_tilesize, h - y_tilesize), index, (x_tilesize, y_tilesize))
            # else:
            slide_sub = slide.read_region((x_wsi, y_wsi), index, (x_tilesize, y_tilesize))

            slide_sub = np.array(slide_sub, dtype=np.uint8)[:, :, :3]
            slide_sub = cv2.resize(slide_sub, (tilesize, tilesize))
        except:
            slide_sub = np.random.randn(tilesize, tilesize, 3)

        return slide_sub

