from Modeling_statistics import correlation, UTest, time_dependence_filter, ElasticNet
from Modeling_ml_surv import Survival_Modeling, Surv_Analysis

from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler

import argparse
import pandas as pd
import numpy as np
import os

# 忽略特定的警告信息
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
# In[]
def load_data(df, df_ft):
    df['ID'] = df['slide_id'].apply(lambda x: x.split('.')[0])
    df_data = pd.merge(df, df_ft, on=['ID'], how='inner')   
    return df_data

def load_datasets(paths, args):
    dit = {}
    for i in range(len(paths)):
        df = pd.read_excel('./data/ciTable/CiTable.xlsx',sheet_name=i)
        df_ft = pd.read_excel(paths[i], sheet_name=0)
        df_data = load_data(df, df_ft)
        ft_names = list(df_data.columns)[11:]
        df_data = df_data.loc[:, ['event', 'survival_months']+ft_names]
        df_data = df_data.rename(columns={args.SURV_RECORD[0]:'event', args.SURV_RECORD[1]:'time'})
        df_data.to_csv(str(i)+'.csv',index=0)
        
        if i==0:
            train_data = df_data.sample(frac=0.8, replace=False, random_state=args.RANDOM_STATE) #,  random_state=42    
            val_data = df_data.drop(train_data.index)
            
            dit['Train'] = train_data
            dit['Val'] = val_data
        else:
            dit[i] = df_data
    return dit

def remix_Train_Val(df0, df1):
    df_data = pd.concat([df0, df1], axis=0)
    train_data = df_data.sample(frac=0.8, replace=False, random_state=args.RANDOM_STATE)    
    val_data = df_data.drop(train_data.index)
    return train_data, val_data

# In[特诊筛选]
def screen_features_pipline(df_data, param):
    ft_names = list(df_data.drop(columns=['event', 'time'], axis=1).columns)
    
    # UTest
    print('UTest', param['p_u'])
    dit_ft_names_filted = {}

    for time_thd in param['timelist']: # only for survival analysis
        # 随访大于最大观测时间且大于一定人数，人数过少影响统计效力
        if df_data['time'].max() > time_thd and df_data[df_data['time']>time_thd].shape[0]>20: 
            df_t = df_data.copy()
            df_t = time_dependence_filter(df_t, time_thd)  
            df_t = df_t.loc[:, ['event']+ft_names]
            remain_names = UTest(df_t, param['p_u'], label_Flag=['event', 1])
            dit_ft_names_filted[time_thd] = remain_names
            
    # 合并同类项    
    l = [] 
    for time_thd in param['timelist']:
        l.extend(dit_ft_names_filted[time_thd])
    l = list(set(l))
    # update table
    ft_names = l
    df_data = df_data.loc[:, ['event', 'time']+ft_names]
    print('After UTest, remain number of features: {}'.format(len(ft_names)))
    
    # Eliminate collinearity
    print('Eliminate collinearity')
    df_t = df_data.copy()
    df_t = df_t.drop(columns=['event', 'time'])
    remove_names = correlation(df_t, thd=param['r_corr'])
    df_t = df_t.drop(columns=remove_names)
    # update
    ft_names = list(df_t.columns)
    df_data = df_data.loc[:, ['event', 'time']+ft_names]
    print('After Eliminate collinearity, remain number of features: {}'.format(len(ft_names)))

    
    # ElasticNet    
    print('ElasticNet', param['coef'])
    dit_remain_names = {}
    coeff = param['coef']
    for time_thd in param['timelist']: # only for survival analysis
        # 随访大于最大观测时间且大于一定人数，人数过少影响统计效力
        if df_data['time'].max() > time_thd and df_data[df_data['time']>time_thd].shape[0]>20: 
            df_t = df_data.copy()
            df_t = time_dependence_filter(df_t, time_thd)  
            df_t = df_t.loc[:, ['event']+ft_names]
            remain_names = ElasticNet(df_t, coeff, label_Flag=['event', 1], shuffle=True, num_cv_fold=5)
            dit_remain_names[time_thd] = remain_names
            print('ElasticNet: coeff={}, time_thd={}, ft_num_remain={}'.format(coeff, time_thd, len(remain_names)))
            
    # 合并同类项    
    l = [] 
    for time_thd in param['timelist']:
        l.extend(dit_remain_names[time_thd])
    ft_names = list(set(l))
    # update table
    df_data = df_data.loc[:, ['event', 'time']+ft_names]
    print('After ElasticNet    , remain number of features: {}'.format(len(ft_names)))
    
    return ft_names

def save_selected_tables(data):
    
    return

def dict_to_str_with_underscore(dictionary):
    # 将字典中的键值对转换为字符串
    items = [f"{key}-{value}" for key, value in dictionary.items()]
    # 使用下划线连接所有的键值对字符串
    result = "_".join(items)
    return result
# In[]
if __name__ == '__main__':
    print('Happy everyday~')
    
    parser = argparse.ArgumentParser(description='PyHistomics_modeling')
    parser.add_argument('--SURV_RECORD', default=['event', 'survival_months'])
    parser.add_argument('--TIME_OBSERVATION', default=[3, 6, 12, 18, 24, 36]) # [3, 6, 12, 18, 24, 36]
    parser.add_argument('--P_THD_UTEST', default=[0.05]) #np.arange(0.001, 0.05, 0.002)  
    parser.add_argument('--R_THD_correlation', default=[0.95]) # correlation thd
    parser.add_argument('--COEFF_ELASTICNET', default=[1]) # np.arange(0.01, 1, 0.02)  
    parser.add_argument('--RANDOM_STATE', default=1)# random_state np.random.randint(12)
    parser.add_argument('--SELECTED_DATA', default='./feature_selected/selected_data.npy')# random_state
    args = parser.parse_args()
    
    paths = ['./results/PyHistomics features_pumch.xlsx',
             './results/PyHistomics features_hbfh.xlsx',
             './results/PyHistomics features_301.xlsx',
             './results/PyHistomics features_tcga.xlsx']
    dit_data = load_datasets(paths, args)    
    
    # 准备特征筛选的参数
    param_list = []
    for p_u in args.P_THD_UTEST:
        for r_p in args.R_THD_correlation:
            for coef in args.COEFF_ELASTICNET:
                param_list.append({'p_u': p_u, 
                                   'timelist': args.TIME_OBSERVATION,
                                   'r_corr': r_p,
                                   'coef': coef,
                                   'randseed': args.RANDOM_STATE,
                                   })
          
    print('# Setting INFO:', args)
    for param in param_list:   
        #以下是特征筛选步骤
        if os.path.exists(args.SELECTED_DATA):
            dit_data = np.load(args.SELECTED_DATA, allow_pickle=True).item()
        else:
            ft_names = screen_features_pipline(dit_data['Train'], param)
            for key, value in dit_data.items():
                dit_data[key] = value.loc[:, ['event', 'time']+ft_names]       
            filename = dict_to_str_with_underscore(param)+'.npy'
            np.save(os.path.join('./feature_selected', filename), dit_data) # 注意带上后缀名
        
    

        # survAny = Surv_Analysis() # 分析结果
        # SM = Survival_Modeling() # 模型
        # SA = Surv_Analysis() # 模型
        # E_flag = 'event'
        # T_flag = 'time'
        # # modeling
        # # alphas = np.arange(0.1, 1.01, 0.1)
        # alphas = [1]
        # for i in range(0,len(alphas)): 
        #     # dit_data['Train'], dit_data['Val'] = remix_Train_Val(dit_data['Train'], dit_data['Val'])        
        #     df_data = dit_data[3].sample(frac=1, replace=False)                
        #     x = df_data.drop(columns=['event', 'time'], axis=1)
        #     x = x.fillna(0)
        #     # scaler = StandardScaler()
        #     # x = scaler.fit_transform(x)
        #     y = Surv.from_dataframe('event', 'time', df_data)   # 'os_event','os_days'
        #     # m = SM.cox_Lasso(x,y,plot_flag=True,l1_value=alphas[i])
            
        #     SA.function_cph(df_data,list(x.columns),['event', 'time'])

        #     break
            
            # for key, value in dit_data.items():    
            #     df_data = dit_data[key].sample(frac=1, replace=False)
                
            #     x = df_data.drop(columns=['event', 'time'], axis=1)
            #     x = x.fillna(0)
            #     y = Surv.from_dataframe('event', 'time', df_data)   # 'os_event','os_days'
            #     # if key == 'Train':
            #     m = best_model.fit(x,y)
            #     # m = best_model
            #     print(np.sum(m.coef_ != 0),'****')
            #     y_score = m.predict(x)
            #     df_re = pd.DataFrame([])
            #     df_re['risk'] = y_score
            #     df_re['event'] = y['event']
            #     df_re['time'] = y['time']
            #     c_harrell = 1-survAny.c_Index(df_re, 'risk', ['time', 'event']) # c-index
            #     print('# Epoch:{}, alpha={}, c-index, set-{}={}'.format(i, alphas[i], key, c_harrell))
            # # except:
            # #     continue
            # print('---------------------------')
  
    
    
