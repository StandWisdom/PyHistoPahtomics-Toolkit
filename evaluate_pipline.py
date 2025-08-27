import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sklearn.metrics import roc_auc_score

import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# In[]
# 构建Transformer模型
class TransformerSurvivalModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerSurvivalModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_encoder_layers
        )
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # 增加一个批次维度以适应Transformer输入
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_projection(x)
        return x
    
# 计算时间依赖AUC
def calculate_auc(risk_scores, time, event, time_points):
    auc_scores = {}
    for tp in time_points:
        labels = (time <= tp).astype(int)
        mask = (time <= tp) | (event == 1)  # 仅考虑截尾或事件发生的样本
        if np.sum(labels[mask]) > 0 and np.sum(1 - labels[mask]) > 0:
            auc = roc_auc_score(labels[mask], risk_scores[mask])
            auc_scores[tp] = auc
        else:
            auc_scores[tp] = None
    return auc_scores 

# In[Data prepare]
parser = argparse.ArgumentParser(description='PyHistomics_modeling')
parser.add_argument('--SELECTED_DATA', 
                    default='./feature_selected/p_u-0.05_timelist-[3, 6, 12, 18, 24, 36]_r_corr-0.95_coef-1_init.npy')# random_state
args = parser.parse_args()
dit_data = np.load(args.SELECTED_DATA, allow_pickle=True).item()

# In[]
# 2. 参数设置
input_dim = 39
d_model = 256
nhead = 8
num_encoder_layers = 8
dim_feedforward = 512
dropout = 0

# 实例化模型
model = TransformerSurvivalModel(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)

# 3. 加载保存的模型权重
model.load_state_dict(torch.load('./checkpoint/transformer_model_pc.pth')) # discovery
print('Model loaded from transformer_model.pth')
model.cuda()
model.eval()

# Eval
eval_list = ['Train', 'Val', 1, 2, 3]
for key in eval_list:
    df = dit_data[key]
    df = df.fillna(0)
    time = np.array(df['time']).astype(np.float32)
    event = np.array(df['event']).astype(bool)
    X = np.array(df.drop(columns=['time', 'event'])).astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)    
    X_tensor = torch.tensor(X).cuda()
    # 预测风险值
    with torch.no_grad():
        # risk_scores = model(X_tensor).cpu().numpy().flatten()
        risk_scores = model(X_tensor)
        # 将风险分数转换为0~1之间的值
        # sigmoid = nn.Sigmoid()
        # risk_scores_test_prob = 1-sigmoid(risk_scores).detach().cpu().flatten().numpy()
        risk_scores_test_prob  = -risk_scores.cpu().numpy().flatten()
        
    dit_data[str(key)+'_score'] = risk_scores_test_prob
    
    # 计算c-index
    c_index = concordance_index_censored(event, time, risk_scores_test_prob)[0]
    
    auc_scores = calculate_auc(risk_scores_test_prob, time, event, [6,12,24])
    print(f'${key} c-index: {c_index:.4f},  auc: {auc_scores[6]:.4f}, {auc_scores[12]:.4f}, {auc_scores[24]:.4f}')
    # aus.append(auc_scores[6])

filename = os.path.split(args.SELECTED_DATA)[-1].split('.npy')[0]+'_pred_pc.npy'
np.save(os.path.join('./results/final_pred', filename), dit_data) # 注意带上后缀名