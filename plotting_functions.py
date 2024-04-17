import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pickle
from torchvision import transforms
import copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
import os
import matplotlib.gridspec as gridspec
import gc
import metrics.multivariate as multiv
import metrics.rank_histogram as rH
import stats.wilcoxon_test as wct

mpl.rcParams['axes.linewidth'] = 2

################################# GRAPHS SETUP
font = {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': 25,
    }

##### ESTHETICS AND TITLE NAMES
base_vars = ['u','v','t2m']
color_p = ['black', 'darkgreen','royalblue', 'red', 'darkorange', 'cyan', 'gold', 'pink', 'tan', 'slategray', 'purple', 'palegreen', 'orchid', 'crimson', 'firebrick']
line = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid','solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid',]
dot = ['dotted', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid','solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid',]

case_name = [['ff=5 (km/h)', 'ff=10 (km/h)', 'ff=15 (km/h)', 'ff=20 (km/h)', 'ff=30 (km/h)', 'ff=40 (km/h)'],
            ['', '', '', '', '', ''], 
            ['t2m=278.15 (K)', 't2m=283.15 (K)', 't2m=288.15 (K)', 't2m=293.15 (K)', 't2m=298.15 (K)', 't2m=303.15 (K)']]

name_thresholds = [['5', '10', '15', '20', '30', '40'],
            ['', '', '', '', '', ''], 
            ['5', '10', '15', '20', '25', '30']]
case_name_thresholds = ['ff' ,'dd', 't2m']
var_names_m = ['ff (m/s)', 'dd (°)', 't2m (K)'  ]
echeance = ['+3H', '', '+9H', '', '+15H', '', '+21H', '', '+27H', '', '+33H', '', '+39H', '', '+45H', '']

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

    for exp_idx, exp in enumerate(experiments):
        mean_bias[exp_idx] = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')
    
    mean_bias_LT = group_by_leadtime(mean_bias, mean_bias_LT, config)

    try:
        significance = np.load(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_decisions_0.npy').squeeze()
    except FileNotFoundError:
        print('computing significance')
        wct.significance(experiments, metric, config)
        significance = np.load(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_decisions_0.npy').squeeze()
    print(significance.shape)
    ################################################ MEAN BIAS  
    for var_idx in range(config['var_number']):
        fig,axs = plt.subplots(figsize = (9,7))
        for exp_idx,exp in enumerate(experiments):
            if exp_idx>0:
                markers_on = significance[var_idx,min(0,exp_idx - 1)].nonzero()[0]
                print('markers_on',markers_on)
                print('significance', significance[var_idx,exp_idx -1])
                plt.plot(np.nanmean(mean_bias_LT[exp_idx,:,:,var_idx], axis = (0,2,3)),
                    label=exp['short_name'], color=color_p[exp_idx],markevery=markers_on, marker="D",linestyle=line[exp_idx], linewidth=3.0)
            else:
                plt.plot(np.nanmean(mean_bias_LT[exp_idx,:,:,var_idx], axis = (0,2,3)),
                    label=exp['short_name'], color=color_p[exp_idx],linestyle=line[exp_idx], linewidth=3.5)
              
        axs.set_xticks(range(len(echeance)))
        axs.set_xticklabels(echeance)
        plt.xticks(fontsize='18')
        axs.tick_params(direction='in', length=12, width=1)

        plt.yticks(fontsize='18')
        plt.ylabel(var_names_m[var_idx], fontsize='18')
        plt.legend(fontsize=10, ncol=1, frameon = False, loc='lower right')
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + case_name_thresholds[var_idx] + '.pdf')
        plt.close()
    
    del mean_bias_LT
    del mean_bias
    gc.collect()

######################## ENSEMBLE CRPS
def plot_ensembleCRPS(experiments, metric, config):
    crps_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], config['var_number']), dtype = ('float32'))
    crps_scores_LT = np.zeros((len(experiments), config['number_dates'],config['lead_times'], config['var_number']), dtype = ('float32'))

    try:
        significance = np.load(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_decisions_0.npy').squeeze()
    except FileNotFoundError:
        print('computing significance')
        wct.significance(experiments, metric, config)
        significance = np.load(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_decisions_0.npy').squeeze()
    print(significance.shape)

    for exp_idx, exp in enumerate(experiments):
        crps = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')
        crps_scores[exp_idx] = crps[:,:,0]
    crps_scores_LT = group_by_leadtime(crps_scores,crps_scores_LT,config)

    for var_idx in range(config['var_number']):
        dist_0 = crps_scores_LT[0,:,0:5,var_idx]
        dist_0 = dist_0.reshape(config['number_dates']*5)
        fig,axs = plt.subplots(figsize = (9,7))        
        for exp_idx, exp in enumerate(experiments[1:]):
            dist = crps_scores_LT[exp_idx + 1,:,0:5,var_idx]
            dist = dist.reshape(config['number_dates']*5)
            axs.hist(dist-dist_0, bins=50)
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name']+ '_diff_'+ case_name_thresholds[var_idx]+'_'+ exp['name']+'.pdf')
        plt.close()

    for var_idx in range(config['var_number']):
        fig,axs = plt.subplots(figsize = (9,7))
        for exp_idx, exp in enumerate(experiments):
            if exp_idx>0:
                markers_on = significance[var_idx,min(0,exp_idx - 1)].nonzero()[0]
                print('markers_on',markers_on)
                print('significance', significance[var_idx,exp_idx -1])
                plt.plot(np.nanmean(crps_scores_LT[exp_idx,:,:,var_idx], axis=(0)),  label=exp['short_name'], 
                color=color_p[exp_idx], linestyle = line[exp_idx], markevery=markers_on, marker="D", linewidth=3.0)
            else:
                plt.plot(np.nanmean(crps_scores_LT[exp_idx,:,:,var_idx], axis=(0)),  label=exp['short_name'],
                 color=color_p[exp_idx], linestyle = line[exp_idx], linewidth=3.0)

        plt.xticks( fontsize ='18')
        axs.set_xticks(range(len(echeance)))
        axs.set_xticklabels(echeance)
        axs.tick_params(direction='in', length=12, width=1)
        plt.yticks(fontsize ='18')
        plt.ylabel(var_names_m[var_idx], fontsize= '18', fontdict=font)
        plt.legend(fontsize=10, ncol=1, frameon=False, loc='lower right')
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + case_name_thresholds[var_idx] +'.pdf')
        plt.close()
    del crps_scores
    del crps_scores_LT
    gc.collect()

######################## ENSEMBLE CRPS
def plot_ensembleCRPSunfair(experiments, metric, config):
    crps_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], config['var_number']), dtype = ('float32'))
    crps_scores_LT = np.zeros((len(experiments), config['number_dates'],config['lead_times'], config['var_number']), dtype = ('float32'))

    try:
        significance = np.load(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_decisions_0.npy').squeeze()
    except FileNotFoundError:
        print('computing significance')
        wct.significance(experiments, metric, config)
        significance = np.load(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_decisions_0.npy').squeeze()
    print(significance.shape)

    for exp_idx, exp in enumerate(experiments):
        crps = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')
        crps_scores[exp_idx] = crps[:,:,0]
    crps_scores_LT = group_by_leadtime(crps_scores,crps_scores_LT,config)

    for var_idx in range(config['var_number']):
        dist_0 = crps_scores_LT[0,:,0:5,var_idx]
        dist_0 = dist_0.reshape(config['number_dates']*5)
        fig,axs = plt.subplots(figsize = (9,7))        
        for exp_idx, exp in enumerate(experiments[1:]):
            dist = crps_scores_LT[exp_idx + 1,:,0:5,var_idx]
            dist = dist.reshape(config['number_dates']*5)
            axs.hist(dist-dist_0, bins=50)
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name']+ '_diff_'+ case_name_thresholds[var_idx]+'_'+ exp['name']+'.pdf')
        plt.close()

        # We can set the number of bins with the *bins* keyword argument.
    for var_idx in range(config['var_number']):
        fig,axs = plt.subplots(figsize = (9,7))
        for exp_idx, exp in enumerate(experiments):                      
            if exp_idx>0:
                markers_on = significance[var_idx,min(0,exp_idx - 1)].nonzero()[0]
                print('markers_on',markers_on)
                print('significance', significance[var_idx,exp_idx -1])
                plt.plot(np.nanmean(crps_scores_LT[exp_idx,:,:,var_idx], axis=(0)),  label=exp['short_name'], 
                color=color_p[exp_idx], linestyle = line[exp_idx], markevery=markers_on,marker="D", linewidth=3.0)
            else:
                plt.plot(np.nanmean(crps_scores_LT[exp_idx,:,:,var_idx], axis=(0)),  label=exp['short_name'],
                 color=color_p[exp_idx], linestyle = line[exp_idx], linewidth=3.0)

        plt.xticks( fontsize ='18')
        axs.set_xticks(range(len(echeance)))
        axs.set_xticklabels(echeance)
        axs.tick_params(direction='in', length=12, width=1)
        plt.yticks(fontsize ='18')
        plt.ylabel(var_names_m[var_idx], fontsize= '18', fontdict=font)
        plt.legend(fontsize=10, ncol=1, frameon=False, loc='lower right')
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + case_name_thresholds[var_idx] +'.pdf')
        plt.close()
    del crps_scores
    del crps_scores_LT
    gc.collect()

###########################" SKILLSPREAD
def plot_skillSpread(experiments, metric, config):
    s_p_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], 2, config['var_number'], config['size_H'], config['size_W']), dtype = ('float32'))
    s_p_scores_LT = np.zeros((len(experiments), config['number_dates'], config['lead_times'], 2, config['var_number'],  config['size_H'], config['size_W']), dtype = ('float32'))

    for exp_idx, exp in enumerate(experiments):
        s_p_scores[exp_idx] = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')
    s_p_scores_LT = group_by_leadtime(s_p_scores, s_p_scores_LT, config)

    try:
        significance_0 = np.load(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_decisions_0.npy').squeeze()
        significance_1 = np.load(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_decisions_1.npy').squeeze()

    except FileNotFoundError:
        print('computing significance')
        wct.significance(experiments, metric, config)
        significance_0 = np.load(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_decisions_0.npy').squeeze()
        significance_1 = np.load(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_decisions_1.npy').squeeze()
    print(significance_0.shape)

    for var_idx in range(config['var_number']):
        fig,axs = plt.subplots(figsize = (9,7))    
        for exp_idx,exp in enumerate(experiments):
            if exp_idx>0:
                markers_on = significance_0[var_idx,min(0,exp_idx - 1)].nonzero()[0]
                print('markers_on',markers_on)
                print('significance', significance_0[var_idx,exp_idx -1])
                plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[exp_idx,:,:,0,var_idx]**2., axis =(0,2,3))), 
                        label=exp['short_name'], markevery=markers_on, marker="D", color=color_p[exp_idx], linestyle=line[exp_idx], linewidth=2.5)
                
                markers_on = significance_1[var_idx,min(0,exp_idx - 1)].nonzero()[0]
                plt.plot(np.nanmean(np.sqrt(np.nanmean(s_p_scores_LT[exp_idx,:,:,1,var_idx], axis =(0))), axis=(-2,-1)),
                        color=color_p[exp_idx], markevery=markers_on, marker="D",linestyle=line[exp_idx], linewidth=3.5)

            else:
                plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[exp_idx,:,:,0,var_idx]**2., axis =(0,2,3))), 
                        label=exp['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx], linewidth=2.5)
                plt.plot(np.nanmean(np.sqrt(np.nanmean(s_p_scores_LT[exp_idx,:,:,1,var_idx], axis =(0))), axis=(-2,-1)),
                        color=color_p[exp_idx], linestyle=line[exp_idx], linewidth=3.5)

        plt.xticks( fontsize ='18')
        axs.set_xticks(range(len(echeance)))
        axs.set_xticklabels(echeance)
        axs.tick_params(direction='in', length=12, width= 1)
        plt.yticks(fontsize ='18')
        plt.ylabel(var_names_m[var_idx], fontsize= '18', fontdict=font)
        plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
        plt.savefig(config['output_plots'] + '/' + metric['folder']+ '/' + metric['name'] + case_name_thresholds[var_idx] +'.pdf')
        plt.close()

    del s_p_scores
    del s_p_scores_LT
    gc.collect()

def plot_brierScore(experiments, metric, config):
    Brier_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], 6, config['var_number'], config['size_H'], config['size_W']), dtype = ('float32'))
    Brier_scores_LT = np.zeros((len(experiments), config['number_dates'], config['lead_times'], 6, config['var_number'],  config['size_H'], config['size_W']), dtype = ('float32'))
    for exp_idx, exp in enumerate(experiments):
        print(exp)
        data = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')
        print(data.shape)
        Brier_scores[exp_idx] = data
        
    Brier_scores_LT = group_by_leadtime(Brier_scores,Brier_scores_LT,config)
    try:
        signs = [np.load(f"{config['output_plots']}/{metric['folder']}/{metric['name']}_decisions_{thr}.npy").squeeze() for thr in range(6)]
    except FileNotFoundError:
        print('computing significance')
        wct.significance(experiments, metric,config)
        signs = [np.load(f"{config['output_plots']}/{metric['folder']}/{metric['name']}_decisions_{thr}.npy").squeeze() for thr in range(6)]

    """for threshold in range(6):
        for var_idx in range(config['var_number']):
            fig,axs = plt.subplots(figsize = (9,7))
            for exp_idx,exp in enumerate(experiments):
                if exp_idx>0:
                    markers_on = signs[threshold][var_idx,min(0,exp_idx - 1)].nonzero()[0]
                    print('markers_on',markers_on)
                    print('significance', signs[threshold][var_idx,exp_idx -1])
                    plt.plot(np.nanmean(Brier_scores_LT[exp_idx,:,:,threshold, var_idx], axis= (0,2,3)),
                    label=exp['short_name'], color=color_p[exp_idx], markevery=markers_on, marker="D", linestyle=line[exp_idx])
                else:
                    plt.plot(np.nanmean(Brier_scores_LT[exp_idx,:,:,threshold, var_idx], axis= (0,2,3)),
                    label=exp['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx])
                
            plt.xticks( fontsize ='18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[var_idx][threshold],fontdict = font)
            plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name']+ '_' + str(threshold) + '_' + case_name_thresholds[var_idx] +'.pdf')
            plt.close()

    for threshold in range(6):
        for var_idx in range(config['var_number']):
            fig,axs = plt.subplots(figsize = (9,7))
            for exp_idx,exp in enumerate(experiments):
                if exp_idx>0:
                    markers_on = signs[threshold][var_idx,min(0,exp_idx - 1)].nonzero()[0]
                    print('markers_on',markers_on)
                    print('significance', signs[threshold][var_idx,exp_idx -1])
                    plt.plot(np.nanmean(Brier_scores_LT[0,:,:,threshold, var_idx], axis= (0,2,3)) - np.nanmean(Brier_scores_LT[exp_idx,:,:,threshold, var_idx], 
                        axis= (0,2,3)),
                        label=exp['short_name'], color=color_p[exp_idx],markevery=markers_on, marker="D",linestyle=line[exp_idx], linewidth=3.0)
                else:
                    plt.plot(np.nanmean(Brier_scores_LT[0,:,:,threshold, var_idx], axis= (0,2,3)) - np.nanmean(Brier_scores_LT[exp_idx,:,:,threshold, var_idx], 
                        axis= (0,2,3)),
                        label=exp['short_name'], color=color_p[exp_idx],linestyle=line[exp_idx], linewidth=3.0)

            plt.xticks( fontsize ='18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[var_idx][threshold],fontdict = font)
            plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' +  metric['name']+ '_diff_' + str(threshold) + '_' + case_name_thresholds[var_idx] +'.pdf')
            plt.close()"""
    
    for var_idx in range(config['var_number']):
        fig,axs = plt.subplots(figsize = (9,7))
        for exp_idx,exp in enumerate(experiments):
            brier_diff = np.zeros((6,))
            markers = []
            for threshold in range(6):
                bss_lt = (np.nanmean(Brier_scores_LT[0,:,:,threshold, var_idx],axis=(0,-2,-1)) - np.nanmean(Brier_scores_LT[exp_idx,:,:,threshold, var_idx],axis=(0,-2,-1))) / np.nanmean(Brier_scores_LT[0,:,:,threshold, var_idx],axis=(0,-2,-1))
                #print(bss_lt.shape)
                condition = wct.decision_leadtimes(bss_lt)
                #print(condition)
                if exp_idx>0 and condition:
                    markers.append(threshold)
                brier_diff[threshold] = (np.nanmean(Brier_scores_LT[0,:,:,threshold, var_idx]) - np.nanmean(Brier_scores_LT[exp_idx,:,:,threshold, var_idx])) / np.nanmean(Brier_scores_LT[0,:,:,threshold, var_idx])
            plt.plot(brier_diff,label=exp['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx], markevery=markers, marker="D", linewidth=3.0)
        axs.set_xticks(range(len(case_name[var_idx])))
        print(case_name[var_idx])
        axs.set_xticklabels(name_thresholds[var_idx], rotation=45)
        axs.tick_params(direction='in', length=12, width=2)
        plt.xticks( fontsize ='18')
        plt.yticks(fontsize ='18')
        plt.title(case_name_thresholds[var_idx],fontdict=font)
        plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_diff_thresholds_' + case_name_thresholds[var_idx] + '.pdf')
        plt.close()

    del Brier_scores
    del Brier_scores_LT
    gc.collect()
    
#### RANK HISTOGRAM

def plot_rankHistogram(experiments, metric, config):

    #rank_histo = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], config['var_number'], config['N_bins_max']))

    N_bins= [exp['N_ens'] + 1 for exp in experiments]

    """for var_idx in range(config['var_number']):
        fig,axs = plt.subplots(figsize = (9,7))
        for exp_idx, exp in enumerate(experiments):
            ind = np.arange(N_bins[exp_idx])
            rank_histo = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')
            print('rankhisto shape', rank_histo.shape)
            bins_local = rank_histo.shape[-1]
            rank_histo_plot = rank_histo[:,var_idx].mean(axis=0)
            if bins_local>17:
                print(rank_histo_plot.shape)
                print(bins_local//17)
                split = np.split(rank_histo_plot,[7,14,21,28,35,42,49,56,63,70,77,84,91,98,105,112])
                print(len(split), split[0].shape)
                rank_histo_plot = np.array([s.sum() for s in split])
                print(rank_histo_plot.shape)
            plt.bar(ind, rank_histo_plot, label=exp['short_name'], alpha=0.5, color=color_p[exp_idx])
        plt.title(var_names_m[var_idx],fontdict=font)
        #plt.xticks( fontsize ='18')
        plt.tick_params(bottom = False, labelbottom = False)
        plt.xlabel('Rank', fontsize= '16')
        plt.ylabel('Number of Observations', fontsize= '16')
        axs.tick_params(length=12, width=1)
        plt.yticks(fontsize ='16')
        plt.legend()
        comp_name = '_'.join([exp['short_name'] for exp in experiments])
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_' + case_name_thresholds[var_idx] + f'_compar_{comp_name}.pdf')

    rank_histo=0
    gc.collect()"""

    for exp_idx, exp in enumerate(experiments):
        for var_idx in range(config['var_number']):
            fig,axs = plt.subplots(figsize = (9,7))
            
            
            rank_histo = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')
            ind = np.arange(rank_histo.shape[-1])
            print('rankhisto shape', rank_histo.shape)
            
            rank_histo_plot = rank_histo[:,var_idx].sum(axis=0)
            deltas = rH.unreliability(rank_histo_plot[np.newaxis,:],config['number_dates'] * config['lead_times'])
            print(var_idx, deltas)
            plt.bar(ind, rank_histo[:,var_idx].mean(axis=0))
            plt.title(f"{exp['short_name']} {var_names_m[var_idx]}",fontdict=font)
            #plt.xticks( fontsize ='18')
            plt.tick_params(bottom = False, labelbottom = False)
            plt.xlabel('Rank', fontsize= '18')
            plt.ylabel('Number of Observations', fontsize= '18')
            axs.tick_params(length=12, width=1)
            plt.yticks(fontsize ='16')
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] + '_' + case_name_thresholds[var_idx] + '_'+ exp['short_name']+'.pdf')

            gc.collect()

##################################################### REL DIAGRAM

def plot_relDiagram(experiments, metric, config):
    
    bins = np.linspace(0, 1, num=11)
    freq_obs = np.zeros((10))

    rel_diag_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], 6, 2, config['var_number'], config['size_H'], config['size_W']))
    for exp_idx, exp in enumerate(experiments):
        rel_diag_scores[exp_idx] = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')

    for threshold in range(6):
        for var_idx in range(config['var_number']):
            fig,axs = plt.subplots(figsize = (9,7))
            for exp_idx, exp in enumerate(experiments):
                O_tr = rel_diag_scores[exp_idx,:-2,threshold,1,var_idx]
                X_prob = rel_diag_scores[exp_idx,:-2,threshold,0,var_idx]
                
                for z in range(bins.shape[0]-1):
                    
                    obs = copy.deepcopy(O_tr[np.where((X_prob >= bins[z]) & (X_prob < bins[z+1]), True, False)])
                    obs = obs[~np.isnan(obs)]
                    freq_obs[z] = obs.sum()/obs.shape[0]
                plt.plot(bins[:-1]+0.05, freq_obs, label=exp['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx])
                    
            plt.plot(bins[:-1]+0.05, bins[:-1]+0.05, label='perfect', color='black', linewidth =3 ) ### I don't remember why I'm adding 0.05
            plt.xticks( fontsize ='18')
            plt.xlabel('forecast probability', fontsize= '18')
            plt.ylabel('observation frequency', fontsize= '18')
            axs.tick_params(direction='in', length=12, width=1)
            plt.yticks(fontsize ='18')
            plt.title(case_name[var_idx][threshold],fontdict = font)
            plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + metric['name'] +'_' + str(threshold) + '_' + case_name_thresholds[var_idx] +'.pdf')
            
    
    rel_diag_scores=0
    gc.collect()

###################################### ROC

def plot_ROC(experiments, metric, config):
    A_ROC = np.zeros((len(experiments),config['var_number'],6))
    A_ROC_skill = np.zeros((len(experiments),config['var_number'],6))
    Hit_rate = np.zeros((17))
    false_alarm = np.zeros((17))
    
    Hit_rate[16]=1
    false_alarm[16]=1
    rel_diag_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], 6, 2, config['var_number'], config['size_H'], config['size_W']))
    
    for exp_idx, exp in enumerate(experiments):
        rel_diag_scores[exp_idx] = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + 'relDiagram.npy') ### ATENTION ROC USES SCORES FROM REL_DIAG_SCORES

    bins_roc = np.array([0.99, 0.93, 0.86, 0.79, 0.72, 0.65, 0.58, 0.51, 0.44, 0.37, 0.3, 0.23, 0.14, 0.07, 0.01])

    for threshold in range(6):
        for var_idx in range(config['var_number']):
            fig,axs = plt.subplots(figsize = (9,7))
            for exp_idx,exp in enumerate(experiments):
                O_tr = rel_diag_scores[exp_idx,:,threshold,1,var_idx]
                X_prob = rel_diag_scores[exp_idx,:,threshold,0,var_idx]   
                print(X_prob.shape, O_tr.shape)
                for z in range(bins_roc.shape[0]):
                    print(z, exp_idx, var_idx, threshold)
                    print("positives")
                    forecast_p = copy.deepcopy(X_prob[np.where((X_prob > bins_roc[z]), True, False)])
                    obs = copy.deepcopy(O_tr[np.where((X_prob > bins_roc[z]), True, False)])
                    obs_w_nan = copy.deepcopy(obs[~np.isnan(obs)])
                    for_w_nan = copy.deepcopy(forecast_p[~np.isnan(obs)])
                    for_w_nan[:] = 1
                    TP = (for_w_nan == obs_w_nan).sum()
                    FP = (for_w_nan != obs_w_nan).sum()
                    
                    print("negatives")
                    forecast_n = copy.deepcopy(X_prob[np.where((X_prob <= bins_roc[z]), True, False)])
                    obs = copy.deepcopy(O_tr[np.where((X_prob <= bins_roc[z]), True, False)])
                    obs_w_nan = copy.deepcopy(obs[~np.isnan(obs)])
                    for_w_nan = copy.deepcopy(forecast_n[~np.isnan(obs)])
                    for_w_nan[:] = 0
                    TN = (for_w_nan == obs_w_nan).sum()
                    FN = (for_w_nan != obs_w_nan).sum() 
                    
                    Hit_rate[z+1]= (TP/(TP+FN))
                    false_alarm[z+1] = (FP/(FP+TN))
                    
                plt.plot(false_alarm, Hit_rate,
                         label=exp['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx])
                print("trapz")
                A_ROC[exp_idx,var_idx,threshold] = np.trapz(Hit_rate, false_alarm)
                A_ROC_skill[exp_idx,var_idx,threshold] = 1 - A_ROC[0,var_idx, threshold] / A_ROC[exp_idx,var_idx,threshold]
                


            plt.xticks( fontsize ='18')
            plt.xlabel('False Alarm Rate', fontsize= '18')
            plt.ylabel('Hit Rate', fontsize= '18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[var_idx][threshold],fontdict = font)
            plt.legend(fontsize = 14,frameon = False, ncol=1)
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' +  metric['name'] +'_' + str(threshold) + '_' + case_name_thresholds[var_idx] +'.pdf')
            plt.close()
    for var_idx in range(config['var_number']):
        fig,axs = plt.subplots(figsize = (9,7))
        for exp_idx,exp in enumerate(experiments)[1:]:
            plt.bar([thr for thr in range(6)], A_ROC_skill[exp_idx,var_idx,thr], color=color_p[exp_idx], label=exp['short_name'])
        plt.xticks( fontsize ='18')
        plt.ylabel('Area under ROC skill', fontsize= '18')
        axs.tick_params(direction='in', length=12, width=2)
        plt.yticks(fontsize ='18')
        plt.legend()
        plt.title(case_name[var_idx][threshold],fontdict = font)
        
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + 'AROCall_' + case_name_thresholds[var_idx] +'.pdf')
        plt.close()
    np.save(config['output_plots'] + '/' + metric['folder'] + '/AROC.npy', A_ROC)
    
def plot_ROCfast(experiments, metric, config):
    A_ROC = np.zeros((len(experiments),config['var_number'],6))
    A_ROC_skill = np.zeros((len(experiments),config['var_number'],6))
    Hit_rate = np.zeros((17))
    false_alarm = np.zeros((17))
    
    Hit_rate[16]=1
    false_alarm[16]=1
    rel_diag_scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], 6, 2, config['var_number'], config['size_H'], config['size_W']))
    
    for exp_idx, exp in enumerate(experiments):
        rel_diag_scores[exp_idx] = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + 'relDiagram.npy',mmap_mode='r+')[:config['number_dates'] * config['lead_times']] ### ATENTION ROC USES SCORES FROM REL_DIAG_SCORES

    bins_roc = np.array([0.99, 0.93, 0.86, 0.79, 0.72, 0.65, 0.58, 0.51, 0.44, 0.37, 0.3, 0.23, 0.14, 0.07, 0.01])

    for threshold in range(6):
        for var_idx in range(config['var_number']):
            fig,axs = plt.subplots(figsize = (9,7))
            for exp_idx,exp in enumerate(experiments):
                print('#' * 80)
                print(exp['short_name'])
                print('#' * 80)

                O_tr = rel_diag_scores[exp_idx,:,threshold,1,var_idx]
                X_prob = rel_diag_scores[exp_idx,:,threshold,0,var_idx]   
                for z in range(bins_roc.shape[0]):
                    print(z, exp_idx, var_idx, threshold)
                    print("positives")
                    indices_p = np.where((X_prob > bins_roc[z]), True, False)
                    forecast_p = (X_prob[indices_p])
                    obs = (O_tr[indices_p])
                    obs_w_nan = (obs[~np.isnan(obs)])
                    for_w_nan = (forecast_p[~np.isnan(obs)])
                    for_w_nan[:] = 1
                    TP = (for_w_nan == obs_w_nan).sum()
                    FP = (for_w_nan != obs_w_nan).sum()
                    
                    print("negatives")
                    #print(forecast_p[0], forecast_p.shape)
                    indices_n = np.logical_not(indices_p)
                    forecast_n = X_prob[indices_n]
                    #print(forecast_n[0], forecast_n.shape)
                    #print(obs_w_nan[0], obs.shape)
                    negobs = O_tr[indices_n]
                    #print(negobs_w_nan[0], negobs_w_nan.shape)
                    negobs_w_nan = (negobs[~np.isnan(negobs)])
                    negfor_w_nan = (forecast_n[~np.isnan(negobs)])
                    negfor_w_nan[:] = 0
                    TN = (negfor_w_nan == negobs_w_nan).sum()
                    FN = (negfor_w_nan != negobs_w_nan).sum() 
                    
                    Hit_rate[z+1]= (TP/(TP+FN))
                    false_alarm[z+1] = (FP/(FP+TN))
                    
                plt.plot(false_alarm, Hit_rate,
                         label=exp['short_name'], color=color_p[exp_idx], linestyle=line[exp_idx])
                print("trapz")
                A_ROC[exp_idx,var_idx,threshold] = np.trapz(Hit_rate, false_alarm)
                print(A_ROC[exp_idx,var_idx,threshold])
                A_ROC_skill[exp_idx,var_idx,threshold] = 1 - A_ROC[0,var_idx, threshold] / A_ROC[exp_idx,var_idx,threshold]
                print(A_ROC_skill[exp_idx,var_idx,threshold])


            plt.xticks( fontsize ='18')
            plt.xlabel('False Alarm Rate', fontsize= '18')
            plt.ylabel('Hit Rate', fontsize= '18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[var_idx][threshold],fontdict = font)
            plt.legend(fontsize = 14,frameon = False, ncol=1)
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' +  metric['name'] +'_' + str(threshold) + '_' + case_name_thresholds[var_idx] +'.pdf')
            plt.close()
    for var_idx in range(config['var_number']):
        fig,axs = plt.subplots(figsize = (9,7))
        for exp_idx,exp in enumerate(experiments[1:]):
            print('#' * 80)
            print(exp['short_name'])
            print('#' * 80)
            print(A_ROC_skill[exp_idx+1,var_idx,:])
            plt.bar([thr - 0.5 + 0.5 * (exp_idx+1) for thr in range(6)], A_ROC_skill[exp_idx+1,var_idx,:], color=color_p[exp_idx+1], label=exp['short_name'])
        plt.xticks( fontsize ='18')
        plt.ylabel('Area under ROC skill', fontsize= '18')
        axs.tick_params(direction='in', length=12, width=2)
        plt.yticks(fontsize ='18')
        plt.legend()
        plt.title(f"AROC {case_name_thresholds[var_idx]}",fontdict = font)
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' + 'AROCall_' + case_name_thresholds[var_idx] +'.pdf')
        plt.close()
    np.save(config['output_plots'] + '/' + metric['folder'] + '/AROC.npy', A_ROC)


def plot_spectralCompute(experiments, metric, config):
    spectral = np.zeros((len(experiments),3, 90))
    for exp_idx, exp in enumerate(experiments):
        spectral[exp_idx] = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')

    scale = np.linspace(2 * np.pi / 2.6, 45 * 256 // 128 * 2 * np.pi / 2.6, 45 * 256 // 128)
    for var_idx in range(config['var_number']):
        fig,axs = plt.subplots(figsize = (9,7))
        for exp_idx, exp in enumerate(experiments):
            plt.plot(scale, spectral[exp_idx][var_idx], label=exp['short_name'], color=color_p[exp_idx],linewidth=2.5, linestyle=line[exp_idx])
        plt.title(f"Power Spectrum of {base_vars[var_idx]}",fontdict=font)
        plt.ylabel(f"Power Spectral Density",fontdict=font)
        plt.xlabel("Scale",fontdict=font)
        plt.xticks(fontsize ='18')
        axs.tick_params(direction='in', length=12, width=2)
        plt.yticks(fontsize ='18')
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize = 14,frameon = False, ncol=1)
        plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' +  metric['name'] + '_' + base_vars[var_idx] +'.pdf')  

def plot_SWD(experiments, metric, config):
    swd = np.zeros((len(experiments),4))
    for exp_idx, exp in enumerate(experiments):
        data = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')[:4]
        print("swd shape", data.shape)
        swd[exp_idx] = data

    Range = ["256", "128", "64", "32"]#,"x5", "avg"]
    
    fig,axs = plt.subplots(figsize = (9,7))
    for exp_idx, exp in enumerate(experiments):
        plt.plot(range(len(Range)), swd[exp_idx], label=exp['short_name'], linewidth=2.5, color=color_p[exp_idx], marker='o')
    plt.title(f"multiscale Sliced Wasserstein Distance",fontdict=font)
    plt.ylabel(f"Distance",fontdict=font)
    plt.xlabel("Resolution (grid points)",fontdict=font)
    plt.xticks(fontsize='18',ticks=range(len(Range)), labels=Range)
    axs.tick_params(direction='in', length=12, width=2)
    plt.yticks(fontsize ='18')
    plt.yscale("log")
    plt.legend(fontsize = 14,frameon = False, ncol=1)
    plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' +  metric['name'] +'.pdf')

def plot_MultivarCorr(experiments, metric, config):
    multivar = np.zeros((len(experiments),3,100,100))
    bins = np.zeros((6,101))
    Xs=['u', 'u', 'v']
    Ys=['v','t2m', 't2m']
    Xsindices = [0,0,1]
    Ysindices = [1,2,2]
    ncouples = 3
    for exp_idx, exp in enumerate(experiments):
        if "AROME" in exp['name']:
            exp_arome = exp_idx
            print('exp_arome', exp_arome)
        else:
            data = pickle.load(open(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.p','rb'))
            print("multivar shape", data.keys(), data['hist'].shape)
            multivar[exp_idx] = data['hist'][1]

            if exp_idx==len(experiments)-1:
                multivar[exp_arome] = data['hist'][0]
                bins = data['bins']
                levels = multiv.define_levels(multivar[exp_arome],5)
                print(levels)

   
    for exp_idx, exp in enumerate(experiments):
        if not "AROME" in exp['name']:
            fig,axs=plt.subplots(1, ncouples, figsize=(4*ncouples,2*ncouples))

            for i in range(ncouples):
                print(multivar[exp_arome][i].min(),multivar[exp_arome][i].max())
                cs=axs[i].contourf(bins[Xsindices[i]][:-1], bins[Ysindices[i]][:-1],np.log10(multivar[exp_arome][i]), cmap='plasma', levels=levels[i])
                axs[i].contour(bins[Xsindices[i]][:-1], bins[Ysindices[i]][:-1], np.log10(multivar[exp_idx][i]),cmap='Greys', levels=levels[i])
                axs[i].set_xlabel(Xs[i], fontsize='large', fontweight='bold')
                axs[i].set_ylabel(Ys[i], fontsize='large', fontweight='bold')
            if i==ncouples-1:
                cbax=fig.add_axes([0.9,0.1,0.02,0.83])
                cb=fig.colorbar(cs, cax=cbax)
                cb.ax.tick_params(labelsize=10)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                cb.set_label('Density (log scale)', fontweight='bold', fontsize='large', rotation=270)
                fig.tight_layout(rect=(0.0,0.0,0.9,0.95))
            plt.savefig(config['output_plots'] + '/' + metric['folder'] + '/' +  metric['name'] + exp['name'] + '.pdf')
            plt.close()
