import os
import pandas as pd
import numpy as np
import math

import cv2 # 图形算子
import skimage.feature as skifeature # GLCM使用

# In[First-order 统计计算]
def TissueArea(x):
    # 计算单个类别的绝对面积
    counts = x.value_counts('pred')
    category = sorted(list(counts.index))
    
    dit = {}
    for i in range(len(category)):
        dit[str(i)+'%all_num'] = counts[category[i]]
    return dit

def Typicality(x):
    # 计算单个类别预测值的均值与方差
    # 均值表征组织典型性, 值越高典型性越强；
    # 方差表示同类组织间的组织稳定度（相似度），值越高，组织间的差异越小
    dit = {}
    counts = x.value_counts('pred')
    category = sorted(list(counts.index))
    for clss in category:        
        x_t = x[x['pred']==clss].loc[:,clss]
        dit['PredValueMean_'+ str(clss)] = np.mean(x_t)
        dit['PredValueStd_'+ str(clss)] = np.std(x_t)  
    return dit

def Proportion_num(x):
    # 计算单个类别占与其他组织间比例
    counts = x.value_counts('pred')
    category = sorted(list(counts.index))

    dit = {}
    for i in range(len(category)):
        for j in range(i+1, len(category)):
            dit[str(i)+'%'+str(j)+'_num'] = counts[category[i]]/counts[category[j]]
    return dit

def Proportion_score(x):
    # 计算单个类别与其他组织总得分间比例
    counts = x.value_counts('pred')
    category = sorted(list(counts.index))

    dit = {}
    for i in range(len(category)):
        for j in range(i+1, len(category)):
            dit[str(i)+'%'+str(j)+'_score'] = np.sum(x[i])/np.sum(x[j])
    return dit

def Proportion_num_global(x):
    # 计算单个类别占WSI全部面积的比例
    counts = x.value_counts('pred')
    category = sorted(list(counts.index))
    
    dit = {}
    for i in range(len(category)):
        dit[str(i)+'%all_num'] = counts[category[i]]/x.shape[0]
    return dit

def Proportion_score_global(x):
    # 计算单个类别的预测得分占WSI全部面区域的比例
    counts = x.value_counts('pred')
    category = sorted(list(counts.index))
    
    dit = {}
    for clss in category:        
        x_t = x[x['pred']==clss].loc[:,clss]
        dit[str(clss)+'%all_score'] = np.sum(x_t)/x.shape[0]
    return dit

def calculate_shannon_entropy(data):
    """
    计算给定数据集的香农熵

    参数：
    data: 包含类别标签的数据集，每个样本应该是一个列表，最后一列为类别标签

    返回值：
    shannon_entropy: 香农熵值
    """
    data = list(data['pred'])
    num_samples = len(data)  # 数据集中样本的总数
    dit = {}

    # 统计每个类别的出现次数
    label_counts = {}
    for sample in data:
        current_label = sample  # 每个样本的最后一列为类别标签
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    # 计算香农熵
    shannon_entropy = 0.0
    for label in label_counts:
        probability = float(label_counts[label]) / num_samples
        shannon_entropy -= probability * math.log(probability, 2)
    dit['shannon_entropy'] = shannon_entropy
    return dit
# In[形态学计算]
def moments(tissuemap):
    '''
    mage moments help you to calculate some features like center of mass of 
    the object, area of the object etc.
    '''
    dit = {}
    tissue_idxs = np.unique(tissuemap)    
    for tissue_idx in tissue_idxs:
        if tissue_idx==0: # For whole foreground, input need img    
            mask = np.where(tissuemap != 0, 1, 0).astype(np.uint8)
        else: # For different tissue
            mask = np.where(tissuemap == tissue_idx, 1, 0).astype(np.uint8)
        
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx = []
        cy = []
        area = []
        for cnt in contours:
            M = cv2.moments(cnt)
            area.append(M['m00'])
            cx.append(int(M['m10']/M['m00'])) # 质心
            cy.append(int(M['m01']/M['m00'])) # 质心
            
        # Area
        dit['moments_{}_area_mean'.format(str(tissue_idx))] = np.mean(area)
        dit['moments_{}_area_std'.format(str(tissue_idx))] = np.std(area)
        dit['moments_{}_area_max'.format(str(tissue_idx))] = np.max(area)
        dit['moments_{}_area_min'.format(str(tissue_idx))] = np.min(area)
            
        # Centroid    
        dit['Centroid_{}_x_mean'.format(str(tissue_idx))] = np.mean(cx)
        dit['Centroid_{}_y_mean'.format(str(tissue_idx))] = np.mean(cy)
        dit['Centroid_{}_x_std'.format(str(tissue_idx))] = np.std(cx)
        dit['Centroid_{}_y_std'.format(str(tissue_idx))] = np.std(cy)
        dit['Centroid_{}_x_max'.format(str(tissue_idx))] = np.max(cx)
        dit['Centroid_{}_y_max'.format(str(tissue_idx))] = np.max(cy)
        dit['Centroid_{}_x_min'.format(str(tissue_idx))] = np.min(cx)
        dit['Centroid_{}_y_min'.format(str(tissue_idx))] = np.min(cy)
    return dit

def contour_perimeter(tissuemap):
    '''
    perimeter
    '''
    dit = {}
    tissue_idxs = np.unique(tissuemap)    
    for tissue_idx in tissue_idxs:
        if tissue_idx==0: # For whole foreground, input need img    
            mask = np.where(tissuemap != 0, 1, 0).astype(np.uint8)
        else: # For different tissue
            mask = np.where(tissuemap == tissue_idx, 1, 0).astype(np.uint8)
        
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = []
        for cnt in contours:
            perimeter.append(cv2.arcLength(cnt,True))
            
        dit['perimeter_{}_mean'.format(str(tissue_idx))] = np.mean(perimeter)
        dit['perimeter_{}_std'.format(str(tissue_idx))] = np.std(perimeter)
        dit['perimeter_{}_max'.format(str(tissue_idx))] = np.max(perimeter)
        dit['perimeter_{}_min'.format(str(tissue_idx))] = np.min(perimeter)
    return dit

def Fitting_Ellipse(tissuemap):
    dit = {}
    tissue_idxs = np.unique(tissuemap)    
    for tissue_idx in tissue_idxs:
        if tissue_idx==0: # For whole foreground, input need img    
            mask = np.where(tissuemap != 0, 1, 0).astype(np.uint8)
        else: # For different tissue
            mask = np.where(tissuemap == tissue_idx, 1, 0).astype(np.uint8)
        
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        w = []
        h = []
        ang = []
        for cnt in contours:
            if len(cnt)>=10:
                ellipse = cv2.fitEllipse(cnt)
                w.append(ellipse[1][0])
                h.append(ellipse[1][1])
                ang.append(ellipse[2])
                
        if np.array(w).size ==0 or np.array(h).size==0:
            dit['Ellipse_w_{}_mean'.format(str(tissue_idx))] = 0
            dit['Ellipse_w_{}_std'.format(str(tissue_idx))] = 0
            dit['Ellipse_w_{}_max'.format(str(tissue_idx))] = 0
            dit['Ellipse_w_{}_min'.format(str(tissue_idx))] = 0
            
            dit['Ellipse_h_{}_mean'.format(str(tissue_idx))] = 0
            dit['Ellipse_h_{}_std'.format(str(tissue_idx))] = 0
            dit['Ellipse_h_{}_max'.format(str(tissue_idx))] = 0
            dit['Ellipse_h_{}_min'.format(str(tissue_idx))] = 0
            
            dit['Ellipse_ang_{}_mean'.format(str(tissue_idx))] = 0
            dit['Ellipse_ang_{}_std'.format(str(tissue_idx))] = 0
            dit['Ellipse_ang_{}_max'.format(str(tissue_idx))] = 0
            dit['Ellipse_ang_{}_min'.format(str(tissue_idx))] = 0
        
        else:
            dit['Ellipse_w_{}_mean'.format(str(tissue_idx))] = np.mean(w)
            dit['Ellipse_w_{}_std'.format(str(tissue_idx))] = np.std(w)
            dit['Ellipse_w_{}_max'.format(str(tissue_idx))] = np.max(w)
            dit['Ellipse_w_{}_min'.format(str(tissue_idx))] = np.min(w)
            
            dit['Ellipse_h_{}_mean'.format(str(tissue_idx))] = np.mean(h)
            dit['Ellipse_h_{}_std'.format(str(tissue_idx))] = np.std(h)
            dit['Ellipse_h_{}_max'.format(str(tissue_idx))] = np.max(h)
            dit['Ellipse_h_{}_min'.format(str(tissue_idx))] = np.min(h)
            
            dit['Ellipse_ang_{}_mean'.format(str(tissue_idx))] = np.mean(ang)
            dit['Ellipse_ang_{}_std'.format(str(tissue_idx))] = np.std(ang)
            dit['Ellipse_ang_{}_max'.format(str(tissue_idx))] = np.max(ang)
            dit['Ellipse_ang_{}_min'.format(str(tissue_idx))] = np.min(ang)
    return dit
        
def Minimum_Enclosing_Circle(tissuemap):
    dit = {}
    tissue_idxs = np.unique(tissuemap)    
    for tissue_idx in tissue_idxs:
        if tissue_idx==0: # For whole foreground, input need img    
            mask = np.where(tissuemap != 0, 1, 0).astype(np.uint8)
        else: # For different tissue
            mask = np.where(tissuemap == tissue_idx, 1, 0).astype(np.uint8)
        
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        r = []
        for cnt in contours:
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            r.append(radius)

        dit['Circle_w_{}_mean'.format(str(tissue_idx))] = np.mean(r)
        dit['Circle_w_{}_std'.format(str(tissue_idx))] = np.std(r)
        dit['Circle_w_{}_max'.format(str(tissue_idx))] = np.max(r)
        dit['Circle_w_{}_min'.format(str(tissue_idx))] = np.min(r)
    return dit
          
def convexity(tissuemap):
    '''
    True: convexity
    return: 凸图形占比, 拟合多边形边数
    '''
    dit = {}
    tissue_idxs = np.unique(tissuemap)    
    for tissue_idx in tissue_idxs:
        if tissue_idx==0: # For whole foreground, input need img    
            mask = np.where(tissuemap != 0, 1, 0).astype(np.uint8)
        else: # For different tissue
            mask = np.where(tissuemap == tissue_idx, 1, 0).astype(np.uint8)
        
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_convexes = []
        shapes = []
        for contour in contours:
            # 对轮廓进行平滑处理
            epsilon = 0.01*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,epsilon,True)
            # 判断凹凸性
            is_convex = cv2.isContourConvex(approx)
            is_convexes.append(is_convex)    
            shapes.append(approx.shape[0])
        dit['convexity_{}'.format(str(tissue_idx))] = np.sum(is_convexes)/len(is_convexes)
        dit['ShapeNum_{}_max'.format(str(tissue_idx))] = np.max(shapes)
        dit['ShapeNum_{}_median'.format(str(tissue_idx))] = np.median(shapes)
        dit['ShapeNum_{}_90percent'.format(str(tissue_idx))] = np.quantile(shapes, 0.90)
        dit['ShapeNum_{}_75percent'.format(str(tissue_idx))] = np.quantile(shapes, 0.75) 
    return  dit          
        

# In[纹理计算]
'''
通常用于低倍率WSI的缩略图计算，配合掩膜使用更加具有针对性
也可以直接用于tissuemap计算，用于近似度量组织间的排列、分布规律
<说明>
对比度：测量灰度共生矩阵的局部变化。
相关性：测量指定像素对的联合概率出现。
平方：提供 GLCM 中元素的平方和。也称为均匀性或角二阶矩。
同质性：测量 GLCM 中元素分布与 GLCM 对角线的接近程度。
'''
def glcm(x, **kwags):
    # Param:
    # x: source image
    # List of pixel pair distance offsets - here 1 in each direction
    # List of pixel pair angles in radians
    angle = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    angle_name = ['0', '90', '180', '270']
    angle_0 = [np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8]
    angle_name_0 = ['45', '135', '225', '315']
    angle = angle + angle_0
    angle_name = angle_name + angle_name_0
    graycom = skifeature.graycomatrix(x, [1], angle, levels=256)
    
    dit = {}
    # Find the GLCM properties
    for i in range(len(angle)):
        dit['glcm_contrast_{}_{}'.format(angle_name[i],kwags['description'])] = skifeature.graycoprops(graycom, 'contrast')[0][i]
        dit['glcm_homogeneity_{}_{}'.format(angle_name[i],kwags['description'])] = skifeature.graycoprops(graycom, 'homogeneity')[0][i] # 均匀性
        dit['glcm_energy_{}_{}'.format(angle_name[i],kwags['description'])] = skifeature.graycoprops(graycom, 'energy')[0][i] # 纹理的灰度变化稳定程度的度量
        dit['glcm_correlation_{}_{}'.format(angle_name[i],kwags['description'])] = skifeature.graycoprops(graycom, 'correlation')[0][i]
        dit['glcm_ASM_{}_{}'.format(angle_name[i],kwags['description'])] = skifeature.graycoprops(graycom, 'ASM')[0][i] # 均匀性
    return dit


def LBP(img, **kwags):
    '''
    input should by cv2
    '''
    # settings for LBP
    radius = 1	# LBP算法中范围半径的取值
    n_points = 8 * radius # 领域像素点数
    lbp = skifeature.local_binary_pattern(img, n_points, radius)
    return lbp

# In[]
if __name__ == '__main__':
    '''
    df_ori = pd.read_excel('./results/pred-matrix/1026004_E5_pred-matrix.xlsx')
    df = df_ori.copy()
    
    # In[]
    # 计算预测类别 argmax
    if not 'pred' in df.columns:
        df['pred'] = np.argmax(np.array(df.iloc[:, 2:]), axis=1)
    
    # 设置得分阈值优化假阳预测, 针对组织分类进行统计特征计算
    for i in range(df.shape[0]):
        if df.loc[i, int(df.loc[i, 'pred'])] > 0.8:
            continue
        else:
            df.loc[i, 'pred'] = None
            
    # 剔除 pred 为 None的预测结果
    df = df.dropna(axis=0).copy()
    
    # Extract features
    
    # input need excel
    dit0 = Typicality(df)
    dit1 = Proportion_num(df)
    dit2 = Proportion_score(df)
    dit3 = Proportion_num_global(df)
    dit4 = Proportion_score_global(df)
    '''
    
    # In[]
    # input need img
    img = cv2.imread('data/tst2.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dit5 = glcm(img, description='origin')
    
    m = LBP(img).astype(np.uint8)
    dit6 = glcm(m, description='lbp')













