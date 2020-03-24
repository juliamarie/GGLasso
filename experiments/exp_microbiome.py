"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from microbiome_helper import load_and_transform, load_tax_data, all_assort_coeff
from gglasso.solver.ext_admm_solver import ext_ADMM_MGL
from gglasso.solver.single_admm_solver import ADMM_SGL

from gglasso.helper.experiment_helper import adjacency_matrix, sparsity, consensus
from gglasso.helper.ext_admm_helper import get_K_identity, check_G, load_G, save_G
from gglasso.helper.model_selection import grid_search, single_range_search, ebic, surface_plot, map_l_to_w

K = 26
reg = 'GGL'

all_csv, S, G, ix_location, ix_exist, p, num_samples = load_and_transform(K, min_inst = 5, compute_G = False)

#save_G('data/slr_results/', G)
G = load_G('data/slr_results/')
check_G(G, p)

#%%


l1 = np.linspace(1, 0.2, 3)
l1 = np.append(l1, np.linspace(0.12, 0.01, 7))
#l1 = np.linspace(0.14, 0.11, 5)
w2 = np.logspace(-1, -3, 4)
#w2 = np.linspace(0.02, 0.01, 5)

AIC, BIC, L1, L2, ix, SP, UQED, sol1 = grid_search(ext_ADMM_MGL, S, num_samples, p, reg, l1 = l1, method = 'eBIC', w2 = w2, G = G)

W1 = map_l_to_w(L1,L2)[0]
W2 = map_l_to_w(L1,L2)[1]

#surface_plot(L1,L2, BIC, save = False)
surface_plot(W1,W2, BIC, save = False)

surface_plot(W1,W2, UQED+1, save = False)


sol1 = sol1['Theta']

#%%

#l1 = np.linspace(0.2, 0.05, 5)
#l1 = 5*np.logspace(-1, -2.5, 6)

sol2, sol3, range_stats = single_range_search(S, l1, num_samples)

#%%

Omega_0 = get_K_identity(p)
lambda1 = 0.12
lambda2 = 0.00346 # w2 = 0.02
sol, info = ext_ADMM_MGL(S, lambda1, lambda2, 'GGL', Omega_0, G, eps_admm = 1e-3, verbose = True)

res_multiple2 = sol['Theta']


#%%
# section for saving results
def save_result(res, typ):
    path = 'data/slr_results/res_' + typ
    K = len(res.keys())
    for k in np.arange(K):
        res_k = pd.DataFrame(res[k], index = all_csv[k].index, columns = all_csv[k].index)
        res_k.to_csv(path + '/theta_' + str(k+1) + ".csv")
    print("All files saved")
    return

save_result(sol1, 'multiple')

for j in np.arange(BIC.shape[0]):
    np.savetxt('data/slr_results/res_multiple/BIC_' + str(j) + '.csv', BIC[j,:,:])
np.savetxt('data/slr_results/res_multiple/AIC.csv', AIC)
np.savetxt('data/slr_results/res_multiple/SP.csv', SP)
np.savetxt('data/slr_results/res_multiple/L1.csv', L1)
np.savetxt('data/slr_results/res_multiple/L2.csv', L2)
np.savetxt('data/slr_results/res_multiple/UQED.csv', UQED)

#save_result(res_multiple2, 'multiple2')

#%%
save_result(sol2, 'single_unif')
save_result(sol3, 'single')

#for j in np.arange(sBIC.shape[0]):
#    np.savetxt('data/slr_results/res_single/BIC_' + str(j) + '.csv', sBIC[j,:,:])
#np.savetxt('data/slr_results/res_single/AIC.csv', sAIC)
#np.savetxt('data/slr_results/res_single/SP.csv', sSP)


#%%
# section for loading results
AIC = np.loadtxt('data/slr_results/res_multiple/AIC.csv')
SP = np.loadtxt('data/slr_results/res_multiple/SP.csv')
L1 = np.loadtxt('data/slr_results/res_multiple/L1.csv')
L2 = np.loadtxt('data/slr_results/res_multiple/L2.csv')

BIC = np.zeros((4, L1.shape[0], L1.shape[1]))
for j in np.arange(4):
    BIC[j,:,:] = np.loadtxt('data/slr_results/res_multiple/BIC_' + str(j) + '.csv')

sol1 = dict()  
for k in np.arange(K):
    #sol1[k] = pd.read_csv('data/slr_results/res_multiple/theta_' + str(k+1) + '.csv', index_col = 0).values
    sol1[k] = pd.read_csv('data/slr_results/res_multiple2/theta_' + str(k+1) + '.csv', index_col = 0).values
    
sol2 = dict()  
for k in np.arange(K):
    sol2[k] = pd.read_csv('data/slr_results/res_single_unif/theta_' + str(k+1) + '.csv', index_col = 0).values
    
sol3 = dict()  
for k in np.arange(K):
    sol3[k] = pd.read_csv('data/slr_results/res_single/theta_' + str(k+1) + '.csv', index_col = 0).values
    
    
#%%
########## EVALUATION ########################    
  
info = pd.DataFrame(index = np.arange(K)+1)
info['samples'] = num_samples
info['OTUs'] = p
info['group entry ratio'] = np.round((G[1,:,:] != -1).sum(axis=0) / (p*(p-1)/2),4)

info['sparsity GGL'] = [np.round(sparsity(sol1[k]), 4) for k in sol1.keys()]
info['sparsity s/u'] = [np.round(sparsity(sol2[k]), 4) for k in sol2.keys()]
info['sparsity s/i'] = [np.round(sparsity(sol3[k]), 4) for k in sol3.keys()]


info.to_csv('data/slr_results/info.csv')

#%% 
nnz1, adj1, val1 = consensus(sol1,G)
nnz2, adj2, val2 = consensus(sol2,G)
nnz3, adj3, val3 = consensus(sol3,G)

consensus_min = 5

(nnz1 >=  consensus_min).sum()
(nnz2 >= consensus_min).sum()
(nnz3 >= consensus_min).sum()


info2 = pd.DataFrame(index = ['GGL', 's/u', 's/i'])
info2['edges within groups'] = [nnz1.sum(), nnz2.sum(), nnz3.sum()]
info2['consensus edges'] = [(nnz1 >=  consensus_min).sum(), (nnz2 >=  consensus_min).sum(), (nnz3 >=  consensus_min).sum()]

info2 = info2.astype(int)
info2.to_csv('data/slr_results/info2.csv')

#%%

fig, axs = plt.subplots(5,6)

for k in np.arange(30):
    ax = axs.ravel()[k]
    
    if k >= K:
        ax.axis('off')
        continue

    d = sol1[k] - sol2[k]
    d = abs(sol1[k]) - abs(sol2[k])
    
    np.fill_diagonal(d,0)
    d = d.ravel()
    d = d[d!=0]
    sns.distplot(d, ax = ax, hist_kws = {'normed':True})
    ax.set_xlim(-0.05,0.05)
    ax.vlines(d.mean(), 0, ax.get_ylim()[1])
    #ax.set_title(f"k = {k}")
    
    
    
plt.figure()
plt.hist(nnz1, 26, density=True, histtype='step', cumulative=True, linestyle = '-', lw = 1.5, label='GGL')
plt.hist(nnz2, 26, density=True, histtype='step', cumulative=True, linestyle = '--', lw = 1.5, label='s/u')
#plt.hist(nnz3, 26, density=True, histtype='step', cumulative=True, label='s/i')
plt.xlabel('Nonzero entries in group')
plt.ylabel('Cumulative density')
plt.legend()
plt.yscale('log')


#%% 
all_tax = load_tax_data(K)


df_assort = pd.DataFrame(index = np.arange(K))

df_assort['GGL'] = all_assort_coeff(sol1, all_csv, all_tax)
df_assort['single/uniform'] = all_assort_coeff(sol2, all_csv, all_tax)
df_assort['single/indv'] = all_assort_coeff(sol3, all_csv, all_tax)

df_assort.to_csv('data/slr_results/assort.csv')