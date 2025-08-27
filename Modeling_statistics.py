# Machine learning
import numpy as np
import pandas as pd
import argparse # 

from scipy.stats import ranksums, pearsonr
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# 忽略特定的警告信息
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# 忽略特定的警告信息
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
# In[filter]
def correlation(df, thd=0.95):
    # 计算特征之间的相关系数和 p 值
    correlation_matrix = df.corr(method='pearson')
    # p_values = df.apply(lambda x: df.apply(lambda y: pearsonr(x, y)[1]))
    
    # 设定相关系数和 p 值的阈值
    correlation_threshold = thd
    p_value_threshold = 0.05
    
    # 找出高度相关且显著的特征对
    features_to_remove = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if (abs(correlation_matrix.iloc[i, j]) > correlation_threshold):
                features_to_remove.add(correlation_matrix.columns[i]) # 优先排除前面的
                # features_to_remove.add(correlation_matrix.columns[j]) # 优先排除后面的, 靠前的特征临床解释性更好
    return features_to_remove
    
def UTest(df, p_thd, label_Flag=None):  # 使用独立样本T检验选取特征，并按P值从小到大排序
    '''
    Wilcoxon 秩和检验（Wilcoxon rank-sum test）的函数。这个检验也被称为 Mann-Whitney U 检验
    Noet:
    label_Flag: [label_name, condition]
    '''
    if label_Flag is None:
        label_Flag = ['label', 1]
        
    df_ = df.copy()
    df_ = df_.fillna(0)  # 消除NaN，防止报错
    df_[label_Flag[0]] = df_[label_Flag[0]].apply(lambda x:1 if x==label_Flag[1] else 0)
    
    df_1 = df_[df_[label_Flag[0]]==1]
    df_0 = df_[df_[label_Flag[0]]==0]
    res = []
    for i in range(df_.shape[1]):
        name = df_.columns[i]
        if not name == label_Flag[0]:
            p_value = ranksums(df_0.loc[:, name], df_1.loc[:, name])[1]  # ttest_ind,ranksums
            res.append([p_value, name])
    df_res = pd.DataFrame(res, columns=['p_value', 'name'])
 
    remain_names = list(df_res[df_res['p_value']<p_thd]['name'])
    return remain_names

def ElasticNet(df, coeff, label_Flag=None, shuffle=True, num_cv_fold=5):
    if label_Flag is None:
        label_Flag = ['label', 1]
    df_ = df.copy()
    if shuffle:
        df_ = df_.sample(frac=1, replace=False)
    df_ = df_.fillna(0)  # 消除NaN，防止报错
    df_[label_Flag[0]] = df_[label_Flag[0]].apply(lambda x:1 if x==label_Flag[1] else 0)        
    enetcv = linear_model.ElasticNetCV(l1_ratio=coeff, tol=1e-5, fit_intercept=True, n_alphas=5000, cv=num_cv_fold)
    
    y = np.array(df_[label_Flag[0]])
    x = np.array(df_.drop(columns=[label_Flag[0]]))
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)
    
    ESC = enetcv.fit(x, y) # 仅用于获取系数
    # print("Optimal regularization parameter : %s" % enetcv.alpha_)
    l = list(enetcv.coef_)
    idx = [i for i in range(len(l)) if l[i] != 0]  
    remain_names = list(df_.columns[idx])
    return remain_names


def concordance_rank():
    return


# In[]
def time_dependence_filter(df, thd_time=12):
    # Follow-up time is longer than thd_time
    df = df[(df['event'] == 0) & (df['time'] > thd_time) | (df['event'] == 1)]
    df.loc[(df['event'] == 1) & (df['time'] > thd_time), 'event'] = 0
    return df
    
# In[]

def load_data(df, df_ft):
    df['ID'] = df['slide_id'].apply(lambda x: x.split('.')[0])
    df_data = pd.merge(df, df_ft, on=['ID'], how='inner')   
    return df_data



if __name__ == '__main__':
    print(0)
    
    parser = argparse.ArgumentParser(description='PyHistomics_modeling')
    parser.add_argument('--SURV_RECORD', default=['event', 'survival_months'])
    parser.add_argument('--TIME_OBSERVATION', default=[3, 6, 12, 18, 24, 36])
    parser.add_argument('--P_THD_UTEST', default=[0.05]) #np.arange(0.001, 0.05, 0.002)  
    args = parser.parse_args()
    
    #
    df = pd.read_excel('./data/ciTable/CiTable.xlsx',sheet_name=0)
    df_ft = pd.read_excel('./results/PyHistomics features_pumch.xlsx',sheet_name=0)
    df_data = load_data(df, df_ft)
    
    ft_names = list(df_data.columns)[11:]
    df_data = df_data.loc[:, ['event', 'survival_months']+ft_names]
    df_data = df_data.rename(columns={'event':'event', 'survival_months':'time'})
    
    dit_ft_names_filted = {}
    # UTest
    for p_thd in args.P_THD_UTEST:
        for time_thd in args.TIME_OBSERVATION: # only for survival analysis
            # 随访大于最大观测时间且大于一定人数，人数过少影响统计效力
            if df_data['time'].max() > time_thd and df_data[df_data['time']>time_thd].shape[0]>20: 
                df_t = df_data.copy()
                df_t = time_dependence_filter(df_t, time_thd)  
                df_t = df_t.loc[:, ['event']+ft_names]
                remain_names = UTest(df_t, 0.05, label_Flag=['event', 1])
                dit_ft_names_filted[time_thd] = remain_names
        
    l = [] # 合并同类项
    for time_thd in args.TIME_OBSERVATION:
        l.extend(dit_ft_names_filted[time_thd])
    l = set(l)

    # update table
    # df = df.loc[:, ['event', 'time']+remain_names]





