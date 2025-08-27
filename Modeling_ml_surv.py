import pandas as pd
import os,glob
import numpy as np

from matplotlib import pyplot as plt

# FOr survival
from lifelines import KaplanMeierFitter,CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test

from sksurv.metrics import (concordance_index_censored,
                            concordance_index_ipcw,
                            cumulative_dynamic_auc)
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 忽略特定的警告信息
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 忽略特定的警告信息
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
# In[]
class Surv_Analysis():
    def __init__(self, **kwags):
        print('#CLASS: Surv_Analysis')

    
    def plot_KM(self, df,fac_name,surv_names):
        '''
        fac_name: factor
        surv_names: [time, event]
        '''
        plt.figure()
        ax = plt.subplot(111)
        
        fac = (df[fac_name]==1)
        T = df[surv_names[0]]*12/365.25
        E = df[surv_names[1]]
        
        kmf_pos = KaplanMeierFitter()
        ax = kmf_pos.fit(T[fac], E[fac], label="pos").plot_survival_function(ax=ax,ci_show=True,
                                                                             show_censors=True)# at_risk_counts=True
        
        kmf_neg = KaplanMeierFitter()
        ax = kmf_neg.fit(T[~fac], E[~fac], label="neg").plot_survival_function(ax=ax,ci_show=True,
                                                                             show_censors=True)# at_risk_counts=True
        # Concat
        add_at_risk_counts(kmf_pos, kmf_neg, ax=ax)
        # log-rank test
        results = logrank_test(T[fac], T[~fac], E[fac], E[~fac], alpha=.99)
    #    results.print_summary()
    
        plt.title(fac_name+'; P-value='+str(results.p_value))
        print('#########{}#############'.format(fac_name))
        return
    
    def function_cph(self, df,fac_name,surv_names):
        '''
        fac_name: 变量
        '''
        df = df.fillna(0)
        # Using Cox Proportional Hazards model
        regression_dataset = df.loc[:,fac_name + surv_names]
        cph = CoxPHFitter()
        cph.fit(regression_dataset,duration_col=surv_names[1],event_col=surv_names[0])
        cph.print_summary()
    #    cph.plot()    
        return cph
    
    def c_Index(self, df, fac_name, surv_names):
        '''
        fac_name: 只接受单变量
        '''
        c_harrell_pc = concordance_index_censored(df[surv_names[0]].astype(bool), df[surv_names[1]], df[fac_name])    
        return c_harrell_pc[0]
    
# In[]
class Survival_Modeling():
    def __init__(self, **kwags):
        print('survival_modeling')
        
        
    def plot_coefficients(self, coefs, n_highlight):
        _, ax = plt.subplots(figsize=(9, 6))
        n_features = coefs.shape[0]
        alphas = coefs.columns
        for row in coefs.itertuples():
            ax.semilogx(alphas, row[1:], ".-", label=row.Index)
    
        alpha_min = alphas.min()
        top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
        for name in top_coefs.index:
            coef = coefs.loc[name, alpha_min]
            plt.text(
                alpha_min, coef, name + "   ",
                horizontalalignment="right",
                verticalalignment="center"
            )
    
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.grid(True)
        ax.set_xlabel("alpha")
        ax.set_ylabel("coefficient")
    
    
    def cox_Rigde(self, Xt,y,plot_flag=True):
        # Ridge_cox
        alphas = 10. ** np.linspace(-4, 4, 50)
        coefficients = {}
        cph = CoxPHSurvivalAnalysis()
        for alpha in alphas:
            cph.set_params(alpha=alpha)
            cph.fit(Xt, y)
            key = round(alpha, 5)
            coefficients[key] = cph.coef_
            
        if plot_flag:    
            coefficients = (pd.DataFrame
                .from_dict(coefficients)
                .set_index(Xt.columns)
                )
            self.plot_coefficients(coefficients, n_highlight=5)
        return cph
    
    def cox_Lasso(self, Xt,y,plot_flag=True,l1_value=1.0):
        print(Xt.shape)
        #lasso-cox
        cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=l1_value, alpha_min_ratio=0.01)
        cox_lasso.fit(Xt, y)
        if plot_flag:
            coefficients_lasso = pd.DataFrame(
                cox_lasso.coef_,
                index=Xt.columns,
                columns=np.round(cox_lasso.alphas_, 5)
            ) 
        self.plot_coefficients(coefficients_lasso, n_highlight=5)
        return cox_lasso
    
    def cox_ElasticNet(self, Xt,y,plot_flag=True,l1_value=0.9):
        # elastic_net
        cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=l1_value, alpha_min_ratio=0.01)
        cox_elastic_net.fit(Xt, y)
        if plot_flag:
            coefficients_elastic_net = pd.DataFrame(
                cox_elastic_net.coef_,
                index=Xt.columns,
                columns=np.round(cox_elastic_net.alphas_, 5)
            )
        
        self.plot_coefficients(coefficients_elastic_net, n_highlight=5)
        return cox_elastic_net
    
    def cv_for_best_model_ElasticNet(self, Xt, y,\
                                     plotFlag1=True,plotFlag2=True,\
                                     l1_value=0.9,n_iter=1000,n_fold=5):
        
        coxnet_pipe = make_pipeline(
            StandardScaler(),
            CoxnetSurvivalAnalysis(l1_ratio=l1_value, alpha_min_ratio=0.01, max_iter=n_iter,\
                                   fit_baseline_model = True)
        )
        warnings.simplefilter("ignore", ConvergenceWarning)
        coxnet_pipe.fit(Xt, y)
        
        estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
        cv = KFold(n_splits=n_fold, shuffle=True, random_state=0) # random_state=0
        gcv = GridSearchCV(
            make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_value,\
                                      alpha_min_ratio=0.1, max_iter=n_iter,\
                                      fit_baseline_model = True)),
            param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
            cv=cv,
            error_score='raise', # default=0.5
            n_jobs=8)
        gcv = gcv.fit(Xt, y)            
        
        if plotFlag1:
            cv_results = pd.DataFrame(gcv.cv_results_)
            
            alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
            mean = cv_results.mean_test_score
            std = cv_results.std_test_score
            
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(alphas, mean)
            ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
            ax.set_xscale("log")
            ax.set_ylabel("concordance index")
            ax.set_xlabel("alpha")
            ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
            ax.axhline(0.5, color="grey", linestyle="--")
            ax.grid(True)
            
        best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]        
        print("Number of non-zero coefficients: {}".format(np.sum(best_model.coef_ != 0)))
        
        if plotFlag2:     
            best_coefs = pd.DataFrame(
                best_model.coef_,
                index=Xt.columns,
                columns=["coefficient"]
            )
            
            non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
            print("Number of non-zero coefficients: {}".format(non_zero))
            
            non_zero_coefs = best_coefs.query("coefficient != 0")
            coef_order = non_zero_coefs.abs().sort_values("coefficient").index
            
            _, ax = plt.subplots(figsize=(6, 8))
            non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
            ax.set_xlabel("coefficient")
            ax.grid(True)
            
        return best_model

# In[]    
if __name__ == '__main__':    
    print(0)
    
    df = pd.read_excel('./data/ciTable/CiTable.xlsx',sheet_name=0)
    
    survAny = Surv_Analysis()    
    survAny.plot_KM(df, 'event', ['survival_months', 'event']) # an example
    survAny.function_cph(df, ['event'], ['survival_days', 'censorship']) # an example
    
    c_harrell = survAny.c_Index(df, 'event', ['survival_days', 'censorship']) # c-index
    
    
    
    
    
    
    
    
    
    