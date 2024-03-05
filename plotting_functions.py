import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pickle
from torchvision import transforms
import copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import matplotlib.gridspec as gridspec
import gc

mpl.rcParams['axes.linewidth'] = 2

################################# GRAPHS SETUP
font = {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': 25,
    }

##### ESTHETICS AND TITLE NAMES
color_p = ['black', 'royalblue', 'darkgreen', 'darkorange', 'red', 'cyan', 'gold', 'pink', 'tan', 'slategray', 'purple', 'palegreen', 'orchid', 'crimson', 'firebrick']
line = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid','solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid',]
case_name = [['ff=5 (km/h)', 'ff=10 (km/h)', 'ff=15 (km/h)', 'ff=20 (km/h)', 'ff=30 (km/h)', 'ff=40 (km/h)'],
            ['', '', '', '', '', ''], 
            ['t2m=278.15 (K)', 't2m=283.15 (K)', 't2m=288.15 (K)', 't2m=293.15 (K)', 't2m=298.15 (K)', 't2m=303.15 (K)']]
case_name_thresholds = ['ff' ,'dd', 't2m']
var_names_m = ['ff (m/s)', 'dd (°)', 't2m (K)'  ]
echeance = ['+3H', '', '+9H', '', '+15H', '', '+21H', '', '+27H', '', '+33H', '', '+39H', '', '+45H', '', '+48H', '', '']

def group_by_leadtime(scores,scores_LT,config):
    D_i = 0
    LT_i = 0
    for timestamp in range(config['number_dates'] * config['lead_times']):        
        scores_LT[:,D_i, LT_i] = scores[:,timestamp]
        LT_i = LT_i + 1
        if LT_i==config['lead_times']: 
            D_i = D_i +1
            LT_i = 0
    return scores_LT

##################################### PLOTTING MEAN BIAS RESULTS ################################
def plot_biasEnsemble(experiments, metric, config):
    mean_bias = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], config['var_number'], config['size_H'], config['size_W']), dtype = ('float32'))
    mean_bias_LT = np.zeros((len(experiments), config['number_dates'], config['lead_times'], config['var_number'],  config['size_H'], config['size_W']), dtype = ('float32'))

    for exp_idx, exp in enumerate(experiments.keys()):
        mean_bias[exp_idx] = np.load(config['expe_folder'] + '/' + exp + '/' + metric['name'] + '.npy')
    
    mean_bias_LT = group_by_leadtime(mean_bias, mean_bias_LT, config)
        
    ################################################ MEAN BIAS    
    for var_idx in range(var_number):
        fig,axs = plt.subplots(figsize = (9,7))
        for exp_idx,exp in enumerate(experiments.keys()):        
            plt.plot(np.nanmean(mean_bias_LT[exp_idx,:,:,var_idx], axis = (0,2,3)), label = experiments[exp]['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx] )
            
            axs.set_xticks(range(len(echeance)))
            axs.set_xticklabels(echeance)
            plt.xticks(fontsize='18')
            axs.tick_params(direction='in', length=12, width=1)

            plt.yticks(fontsize='18')
            #plt.title(var_names[i] ,fontdict = font)
            #plt.text(0.6, 0.9, var_names[i] + ' ' + domain[ii],
            #        fontdict = font, transform=axs.transAxes)
            plt.ylabel(var_names_m[var_idx], fontsize='18')
            plt.legend(fontsize=10, ncol=1, frameon = False, loc='lower right')
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + str(var_names_m[var_idx]) + '.pdf')
            plt.close()
    
    del mean_bias_LT
    del mean_bias
    gc.collect()

######################## ENSEMBLE CRPS
def plot_ensembleCRPS(experiments, metric, config):
    crps_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], config['var_number']), dtype = ('float32'))
    crps_scores_LT = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], config['var_number']), dtype = ('float32'))

    for exp_idx, exp in enumerate(experiments.keys()):
        crps = np.load(config['expe_folder'] + '/' + exp + '/' + metric['name'] + '.npy')
        crps_scores[exp_idx] = crps[:,:,0]
    crps_scores_LT = group_by_leadtime(crps_scores,crps_scores_LT,config)

    for var_idx in range(config['var_number']):
        dist_0 = crps_scores_LT[0,:,0:5,var_idx]
        dist_0 = dist_0.reshape(config['number_dates']*5)
        fig,axs = plt.subplots(figsize = (9,7))        
        for exp_idx, exp in enumerate(experiments.keys()[1:]:
            dist = crps_scores_LT[exp_idx,:,0:5,var_idx]
            dist = dist.reshape(config['number_dates']*5)
            axs.hist(dist-dist_0, bins=50)
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name']+ '_diff_'+str(var_names_m[var_idx]))+'_'+str(exp) +'.pdf')
            plt.close()

        # We can set the number of bins with the *bins* keyword argument.
    for var_idx in range(config['var_number']):
        fig,axs = plt.subplots(figsize = (9,7))
        for exp_idx, exp in enumerate(experiments.keys()):                      

            plt.plot(np.nanmean(crps_scores_LT[exp_idx,:,:,var_idx], axis=(0)), label=experiments[exp]['short_name'], color=color_p[exp_idx], linestyle = line[exp_idx])

        plt.xticks( fontsize ='18')
        axs.set_xticks(range(len(echeance)))
        axs.set_xticklabels(echeance)
        axs.tick_params(direction='in', length=12, width=1)
        plt.yticks(fontsize ='18')
        plt.ylabel(var_names_m[var_idx], fontsize= '18', fontdict=font)
        #plt.title(var_names[i] + ' ' + domain[ii],fontdict = font)
        #plt.title(var_names[i] ,fontdict = font)                
        #plt.text(0.6, 0.9, var_names[i] + ' ' + domain[ii],
        #        fontdict = font, transform=axs.transAxes)
        plt.legend(fontsize=10, ncol=1, frameon=False, loc='lower right')
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + str(var_names_m[var_idx]) +'.pdf')
        plt.close()
    del crps_scores
    del crps_scores_LT
    gc.collect()

###########################" SKILLSPREAD
def plot_skillSpread(experiments, metric, config):
    s_p_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], 2, config['var_number'], config['size_H'], config['size_W']), dtype = ('float32'))
    s_p_scores_LT = np.zeros((len(experiments), config['number_dates'], config['lead_times'], 2, config['var_number'],  config['size_H'], config['size_W']), dtype = ('float32'))

    for exp_idx, exp in enumerate(experiments.keys()):
        s_p_scores[exp_idx] = np.load(config['expe_folder'] + '/' + exp + '/' + metric['name'] + '.npy')
    s_p_scores_LT = group_by_leadtime(s_p_scores, s_p_scores_LT, config)

    for var_idx in range(config['var_number']):
        fig,axs = plt.subplots(figsize = (9,7))    
            for exp_idx,exp in enumerate(experiments.keys()):
                plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[exp_idx,:,:,0,var_idx]**2., axis =(0,2,3))), 
                        label=experiments[exp]['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx], linewidth=2)
                plt.plot(np.nanmean(np.sqrt(np.nanmean(s_p_scores_LT[exp_idx,:,:,1,var_idx], axis =(0))), axis=(-2,-1)),
                        label=experiments[exp]['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx] )

                plt.xticks( fontsize ='18')
                axs.set_xticks(range(len(echeance)))
                axs.set_xticklabels(echeance)
                axs.tick_params(direction='in', length=12, width= 1)
                plt.yticks(fontsize ='18')
                plt.ylabel(var_names_m[var_idx], fontsize= '18', fontdict=font)
                plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
                plt.savefig(config['output_plots'] + '/' + metric['folder']+ '/' + metric['name'] + str(var_names_m[var_idx]) +'.pdf')
            plt.close()

    del s_p_scores
    del s_p_scores_LT
    gc.collect()

def plot_brierScore(experiments, metric, config):
    Brier_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], 6, config['var_number'], config['size_H'], config['size_W']), dtype = ('float32'))
    Brier_scores_LT = np.zeros((len(experiments), config['number_dates'], config['lead_times'], 6, config['var_number'],  config['size_H'], config['size_W']), dtype = ('float32'))

    for exp_idx, exp in enumerate(experiments.keys()):
        Brier_scores[exp_idx] = np.load(config['expe_folder'] + '/' + exp + '/' + metric['name'] + '.npy')
        
    Brier_scores_LT = group_by_leadtime(Brier_scores,Brier_scores_LT,config)
        
    for threshold in range(6):
        for var_idx in range(config['var_number']):
            fig,axs = plt.subplots(figsize = (9,7))
            for exp_idx,exp in enumerate(experiments.keys()):
                plt.plot(np.nanmean(Brier_scores_LT[exp_idx,:,:,threshold, var_idx], axis= (0,2,3)),
                label=experiments[exp]['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx])
                
            plt.xticks( fontsize ='18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[var_idx][threshold],fontdict = font)
            plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name']+ '_' + str(threshold) + '_' + str(var_names_m[var_idx]) +'.pdf')
            plt.close()

    for threshold in range(6):
        
        for var_idx in range(config['var_number']):
            fig,axs = plt.subplots(figsize = (9,7))
            for exp_idx,exp in enumerate(experiments.keys()):
                #plt.plot(np.nanmean(Brier_scores_LT[0,:,:,i, j], axis= (0,2,3))-np.nanmean(Brier_scores_LT[k,:,:,i, j], axis= (0,2,3)), label = cases_clean[k], color = color_p[k], linestyle = line[k])
                plt.plot(np.nanmean(Brier_scores_LT[0,:,:,threshold, var_idx] - Brier_scores_LT[exp_idx,:,:,threshold, var_idx], axis= (0,2,3)),
                        label=experiments[exp]['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx])

            plt.xticks( fontsize ='18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[var_idx][threshold],fontdict = font)
            plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' +  metric['name']+ '_diff_' + str(i) + '_' + str(j) +'.pdf')
            plt.close()
    
    for var_idx in range(config['var_number']):
        for exp_idx,exp in enumerate(experiments.keys()):
            brier_diff = np.zeros((6,))
            for threshold in range(6):
                brier_diff[threshold] = np.nanmean(Brier_scores_LT[0,:,:,threshold, var_idx] - Brier_scores_LT[exp_idx,:,:,threshold, var_idx])
            plt.plot(brier_diff,label=experiments[exp]['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx])
        plt.xticks( fontsize ='18')
        axs.set_xticks(range(len(case_name[var_idx])))
        axs.set_xticklabels(case_name[var_idx])
        axs.tick_params(direction='in', length=12, width=2)
        plt.yticks(fontsize ='18')
        plt.title(case_name_thresholds[var_idx],fontdict=font)
        plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_diff_thresholds_' + str(var_names[var_idx]) + '.pdf')
        plt.close()

    del Brier_scores
    del Brier_scores_LT
    gc.collect()
    
#### RANK HISTOGRAM

def plot_rankHistogram(experiments, metric, config)
    rank_histo = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], config['var_number'], config['N_bins_max']))

    for exp_idx, exp in enumerate(experiments.keys()):
        rank_histo[exp_idx] = np.load(config['expe_folder'] + '/' + exp + '/' + metric['name'] + '.npy')


    N_bins= [config['experiments'][0]['dataloaders']['N_ens'] + 1 for k in range(len_tests)]

    for var_idx in range(config['var_number']):
        for exp_idx, exp in enumerate(experiments.keys()):
            fig,axs = plt.subplots(figsize = (9,7))
            ind = np.arange(N_bins[exp_idx])
            print(rank_histo[exp_idx,:,var_idx,0:N_bins[exp_idx]].sum(axis=0).shape)
            plt.bar(ind, rank_histo[exp_idx,:,var_idx,0:N_bins[exp_idx]].sum(axis=0))
            plt.title(cases_clean[exp_idx] + ' ' + var_names_m[var_idx],fontdict=font)
            #plt.xticks( fontsize ='18')
            plt.tick_params(bottom = False, labelbottom = False)
            plt.xlabel('Bins', fontsize= '18')
            plt.ylabel('Number of Observations', fontsize= '18')
            axs.tick_params(length=12, width=1)
            plt.yticks(fontsize ='18')

            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_' + str(var_names_m[var_idx]) + '_' + str(exp) +'.pdf')

    rank_histo=0
    gc.collect()

##################################################### REL DIAGRAM

def plot_relDiagram(experiments, metric, config):
    
    bins = np.linspace(0, 1, num=11)
    freq_obs = np.zeros((10))

    rel_diag_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], 6, 2, config['var_number'], config['size_H'], config['size_W']))
    for exp_idx, exp in enumerate(experiments.keys()):
        rel_diag_scores[exp_idx] = np.load(config['expe_folder'] + '/' + exp + '/' + metric['name'] + '.npy')

    for threshold in range(6):
        for var_idx in range(config['var_number']):
            fig,axs = plt.subplots(figsize = (9,7))
            for exp_idx, exp in enumerate(experiments.keys()):
                O_tr = rel_diag_scores[exp_idx,:-2,threshold,1,var_idx]
                X_prob = rel_diag_scores[exp_idx,:-2,threshold,0,var_idx]
                
                for z in range(bins.shape[0]-1):
                    
                    obs = copy.deepcopy(O_tr[np.where((X_prob >= bins[z]) & (X_prob < bins[z+1]), True, False)])
                    obs = obs[~np.isnan(obs)]
                    print(obs.shape, var_idx)
                    freq_obs[z] = obs.sum()/obs.shape[0]
                plt.plot(bins[:-1]+0.05, freq_obs, label = cases_clean[exp_idx], color = color_p[exp_idx], linestyle = line[exp_idx])
                    
            plt.plot(bins[:-1]+0.05, bins[:-1]+0.05, label = 'perfect', color = 'black', linewidth =3 ) ### I don't remember why I'm adding 0.05
            plt.xticks( fontsize ='18')
            plt.xlabel('forecast probability', fontsize= '18')
            plt.ylabel('observation frequency', fontsize= '18')
            axs.tick_params(direction='in', length=12, width=1)
            plt.yticks(fontsize ='18')
            plt.title(case_name[var_idx][threshold],fontdict = font)
            plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] +'_' + str(threshold) + '_' + var_names_m[var_idx] +'.pdf')
            
    
    rel_diag_scores=0
    gc.collect()

###################################### ROC

def plot_ROC(experiments, metric, config):
    A_ROC = np.zeros((len(experiments)))
    A_ROC_skill = np.zeros((len(experiments)))
    Hit_rate = np.zeros((17))
    false_alarm = np.zeros((17))
    
    Hit_rate[16]=1
    false_alarm[16]=1
    rel_diag_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], 6, 2, config['var_number'], config['size_H'], config['size_W']))
    
    for exp_idx, exp in enumerate(experiments.keys()):
        rel_diag_scores[exp_idx] = np.load(config['expe_folder'] + '/' + exp + '/' + 'relDiagram.npy') ### ATENTION ROC USES SCORES FROM REL_DIAG_SCORES

    bins_roc = np.array([0.99, 0.93, 0.86, 0.79, 0.72, 0.65, 0.58, 0.51, 0.44, 0.37, 0.3, 0.23, 0.14, 0.07, 0.01])

    for threshold in range(6):
        for var_idx in range(var_number):
            fig,axs = plt.subplots(figsize = (9,7))
            for k in range(len_tests):
                O_tr = rel_diag_scores[exp_idx,:-2,threshold,1,var_idx]
                X_prob = rel_diag_scores[exp_idx,:-2,threshold,0,var_idx]   

                for z in range(bins_roc.shape[0]):
                    forecast_p = copy.deepcopy(X_prob[np.where((X_prob > bins_roc[z]), True, False)])
                    obs = copy.deepcopy(O_tr[np.where((X_prob > bins_roc[z]), True, False)])
                    obs_w_nan = copy.deepcopy(obs[~np.isnan(obs)])
                    for_w_nan = copy.deepcopy(forecast_p[~np.isnan(obs)])
                    for_w_nan[:] = 1
                    TP = (for_w_nan == obs_w_nan).sum()
                    FP = (for_w_nan != obs_w_nan).sum()
                    
                    
                    forecast_n = copy.deepcopy(X_prob[np.where((X_prob <= bins_roc[z]), True, False)])
                    obs = copy.deepcopy(O_tr[np.where((X_prob <= bins_roc[z]), True, False)])
                    obs_w_nan = copy.deepcopy(obs[~np.isnan(obs)])
                    for_w_nan = copy.deepcopy(forecast_n[~np.isnan(obs)])
                    for_w_nan[:] = 0
                    TN = (for_w_nan == obs_w_nan).sum()
                    FN = (for_w_nan != obs_w_nan).sum() 
                    
                    Hit_rate[z+1]= (TP/(TP+FN))
                    false_alarm[z+1] = (FP/(FP+TN))
                    
                plt.plot(false_alarm, Hit_rate, label = cases_clean[exp_idx], color = color_p[exp_idx], linestyle = line[exp_idx])

                A_ROC[k] = np.trapz(Hit_rate, false_alarm)
                    
                A_ROC_skill[k]=1-A_ROC[0]/A_ROC[k]
                


            plt.xticks( fontsize ='18')
            plt.xlabel('False Alarm Rate', fontsize= '18')
            plt.ylabel('Hit Rate', fontsize= '18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[var_idx][threshold],fontdict = font)
            plt.legend(fontsize = 14,frameon = False, ncol=1)
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' +  metric['name'] +'_' + str(threshold) + '_' + var_names_m[var_idx] +'.pdf')

            fig,axs = plt.subplots(figsize = (9,7))
            plt.bar(cases_clean[1::], A_ROC_skill[1::])
            plt.xticks( fontsize ='18')
            plt.ylabel('Area under ROC skill', fontsize= '18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[var_idx][threshold],fontdict = font)
            
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_' + str(threshold) + '_' + var_names_m[var_idx] +'.pdf')