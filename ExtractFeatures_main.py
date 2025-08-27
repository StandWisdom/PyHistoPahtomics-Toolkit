from PyHistomics import *
from preprocessing import PreOptWSI

#
import os
import numpy as np
import pandas as pd
import glob
import argparse # 
import platform
import cv2
import einops

# Openslide
if 'Win' in platform.platform(): 
    OPENSLIDE_PATH = r'E:\OfficialToolKit\openslide-win64-20230414\bin'
    os.add_dll_directory(OPENSLIDE_PATH)
import openslide

from torchvision import transforms
import torchstain
# In[excel]
def preproc_df(df):
    # 设置得分阈值优化假阳预测, 针对组织分类进行统计特征计算
    for i in range(df.shape[0]):
        if df.loc[i, int(df.loc[i, 'pred'])] > 0.8:
            continue
        else:
            df.loc[i, 'pred'] = None
    # 剔除 pred 为 None的预测结果
    df = df.dropna(axis=0).copy()
    return df

def prepare_excel(path):
    df_ori = pd.read_excel(path)
    df = df_ori.copy()
    
    # 计算预测类别 argmax
    if not 'pred' in df.columns:
        df['pred'] = np.argmax(np.array(df.iloc[:, 2:]), axis=1)
    return df

# In[]
class Handle_WSI():
    def __init__(self):
        print('# class: Handle_WSI')
        
    
    def generate_tissuemap(self, PATH):
        result_df = pd.read_excel(PATH)
        xy = np.array(result_df.iloc[:, :2])
        pred = np.array(result_df.iloc[:, 2:])
    
        tilesize = 256
        step = 1
        wsi_pyramid = [2, 4, 8]
        ratio = tilesize * step * wsi_pyramid[0]
        
        h_ = np.min(xy[:, 1])
        w_ = np.min(xy[:, 0])
        self.h_min = h_
        self.w_min = w_
        self.h_max = np.max(xy[:, 1])
        self.w_max = np.max(xy[:, 0])
        h = int((np.max(xy[:, 1])-h_) // ratio + 1)
        w = int((np.max(xy[:, 0])-w_) // ratio + 1)
        canvas = np.zeros([h, w])
    
        for i in range(xy.shape[0]):
            x = (xy[i, 1]-h_) // ratio
            y = (xy[i, 0]-w_) // ratio
            pred_ = pred[i, :]
            label = np.argmax(pred_)
    
            canvas[x, y] = label+1
        return canvas.astype(np.uint8)
    
    def WSI_thumbnail(self, path, mag, **kwags):
        args = kwags['keywords']
        # openslide 
        slide = openslide.OpenSlide(path)
        dim = list(slide.level_dimensions)
        dsample = list(slide.level_downsamples)  # 每一个级别K的对应的下采样因子，下采样因子应该对应一个倍率
        
        # resolution norm
        mpp = [slide.properties['openslide.mpp-x'],slide.properties['openslide.mpp-y']] # mpp
        mpp = np.around(np.array(mpp).astype(float), 6)# mpp
        
        # size
        arr = np.abs(np.array(dsample)-mag)
        index = arr.argmin()
        x_tilesize_rate = (mpp[0] * dsample[index]) / (args.STDMPP[0] * mag) # w
        y_tilesize_rate = (mpp[1] * dsample[index]) / (args.STDMPP[1] * mag)# h
        
        # origin
        img_thumbnail = slide.get_thumbnail(dim[index])
        img_thumbnail = np.array(img_thumbnail)
        # crop by foreground map
        img_thumbnail = img_thumbnail[round(self.h_min/dsample[index]):int(self.h_max/dsample[index]), 
                                      round(self.w_min/dsample[index]):int(self.w_max/dsample[index])]
        
        # resolution norm
        x_tilesize = np.around(img_thumbnail.shape[1] * x_tilesize_rate).astype(np.uint32)
        y_tilesize = np.around(img_thumbnail.shape[0] * y_tilesize_rate).astype(np.uint32)    
        img_thumbnail_ = cv2.resize(img_thumbnail, (x_tilesize, y_tilesize))
        
        return img_thumbnail_
    
    def Norm_wsi(self, img_thumbnail):
        preopt = PreOptWSI()
        
        magnification_otsu = 1
        img_thumbnail = cv2.resize(img_thumbnail, (img_thumbnail.shape[1]*magnification_otsu, img_thumbnail.shape[0]*magnification_otsu))
        foreground = preopt.seg_threshold(img_thumbnail)
        foreground = preopt.FindThresholdRegion(foreground, 0.2)  # find >thd region
        img_thumbnail_ = cv2.bitwise_and(img_thumbnail, img_thumbnail, mask = foreground)
        img_thumbnail_[np.where((img_thumbnail_ ==[0,0,0]).all(axis=2))]=[255,255,255]
    
        foreground = 1-preopt.seg_otsu(img_thumbnail_) # 1-背景
        preopt.FindMaxRegion(foreground)
        
        return

    def color_norm(self, img, **kwags):
        # st = time.time()
        target = cv2.resize(cv2.cvtColor(cv2.imread("./data/template.png"), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))
        to_transform = img

        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255)
        ])
        
        torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        torch_normalizer.fit(T(target))        
        t_to_transform = T(to_transform)
        norm, H, E = torch_normalizer.normalize(I=t_to_transform, stains=True)
        if kwags['variable_type'] == 'numpy':
            return norm.cpu().numpy(), H.cpu().numpy(), E.cpu().numpy()
        else:
            return norm, H, E
# In[]

def extract_features(args):
    dict_all = {} 
    # Load data
    df = prepare_excel(args.EXCELPATH)
    name = os.path.split(args.EXCELPATH)[-1].split('_pred-matrix.xlsx')[0]  
    srcDir = args.WSIDIR
    datalist = glob.glob(srcDir+'/*.svs', recursive=True) + \
                glob.glob(srcDir + '/*.ndpi', recursive=True)

    # print(datalist)
    for path in datalist:
        if name in path:
            path_wsi = path
            break
        else:
            path_wsi = None
            continue
    if path_wsi is None:
        print('Not found WSI file.',name)
        return None
    print('***',path_wsi)
        
    # first-order features, Extract features, input need excel
    dict_all.update(TissueArea(df))
    dict_all.update(Typicality(df))
    dict_all.update(Proportion_num(df))
    dict_all.update(Proportion_score(df))
    dict_all.update(Proportion_num_global(df))
    dict_all.update(Proportion_score_global(df))
    dict_all.update(calculate_shannon_entropy(df))
    
    # texture features
    hwsi = Handle_WSI()
    tissuemap = hwsi.generate_tissuemap(args.EXCELPATH)
    for mag in args.WSIMAG:
        img = hwsi.WSI_thumbnail(path_wsi, mag, keywords=args)    
        tissuemap = cv2.resize(tissuemap, (img.shape[1], img.shape[0]),interpolation=cv2.INTER_NEAREST) # 不能直接resize    
        
        #
        img, H, E =hwsi.color_norm(img, variable_type='numpy')   
        tissue_idxs = np.unique(tissuemap)    
        for tissue_idx in tissue_idxs:
            if tissue_idx==0: # For whole foreground, input need img    
                mask = np.where(tissuemap != 0, 1, 0).astype(np.uint8)
            else: # For different tissue
                mask = np.where(tissuemap == tissue_idx, 1, 0).astype(np.uint8)
            
            img = img.astype(np.uint8)
            img_ = cv2.bitwise_and(img, img, mask=mask)
            img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            dict_all.update(glcm(img_gray, description='origin_'+str(mag)+'_'+str(tissue_idx))) # texture features
            m = LBP(img_gray).astype(np.uint8)
            dict_all.update(glcm(m, description='lbp_'+str(mag)+'_'+str(tissue_idx))) # texture features

    # shape
    dict_all.update(moments(tissuemap)) # 面积 质心
    dict_all.update(contour_perimeter(tissuemap)) # 轮廓 周长
    dict_all.update(Fitting_Ellipse(tissuemap)) # 椭圆拟合
    dict_all.update(Minimum_Enclosing_Circle(tissuemap)) # 最小外接圆
    dict_all.update(convexity(tissuemap)) # 凸性
    return dict_all

def aggregate_excel(saveDir):
    ftlist = glob.glob(saveDir+'/*.xlsx', recursive=True)
    df_all = pd.DataFrame([])
    ids = []
    for i in range(len(ftlist)):
        df = pd.read_excel(ftlist[i])
        df_all = pd.concat([df_all, df[0]], axis=1)
        ids.append( os.path.split(ftlist[i])[-1].split('_PyHistomics_features.xlsx')[0] )
    
    df_all = df_all.T
    df_all.columns = list(df['item'])
    df_all.insert(0, 'ID', ids)
    df_all = df_all.reset_index(drop=True)   
    
    name = saveDir.split('/')[-1]+'.xlsx'
    df_all.to_excel(os.path.join('./results',name), index=0)
    return

# In[]
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    
    excelDir = r'./results/hbfh'
    wsiDir = r'L:\pancreatic cancer\hbfh' # openslide路径不能含中文
    saveDir = './results/PyHistomics features_hbfh'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    excellist = glob.glob(excelDir+'/*.xlsx', recursive=True)
    
    for excelpath in excellist[0:]:
        print(excelpath)
        # Init
        parser = argparse.ArgumentParser(description='PyHistomics')
        parser.add_argument('--WSIDIR', default=wsiDir)
        parser.add_argument('--EXCELPATH', default=excelpath)
        parser.add_argument('--WSIMAG', default=[64, 128]) # 128: 32 um/pixel 需要和tissuemap的分辨率匹配
        parser.add_argument('--STDMPP', default=[0.25, 0.25])
        args = parser.parse_args()
        # extract features
        dict_all = extract_features(args)
        # save 
        df_save = pd.DataFrame.from_dict(dict_all, orient='index')  
        df_save['item'] = df_save.index
        df_save = df_save.reset_index(drop=True)
        name = os.path.split(excelpath)[-1].split('_pred-matrix.xlsx')[0]  
        savepath = os.path.join(saveDir, name+'_PyHistomics_features.xlsx')
        df_save.to_excel(savepath, index=0)
        
    # aggregate results
    aggregate_excel(saveDir)
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

