
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
import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import yaml

from core.experiment_set import ExperimentSet

mpl.rcParams['axes.linewidth'] = 2

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data



def setup_logger(debug=False):
    """
    Configure a logger with specified console and file handlers.
    Args:
        debug (bool): Whether to enable debug logging.
        log_file (str): The name of the log file.
    Returns:
        logging.Logger: The configured logger.
    """
    console_format = '%(asctime)s - %(levelname)s - %(message)s'

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_formatter = logging.Formatter(console_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Plot experiments.")
    parser.add_argument("--config", type=str, default="plotting.yml", help="Path to the configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Configure logger
    logger = setup_logger(debug=args.debug)

    # Log the start of the program
    logger.info("Starting program.")


    try:
        logger.info(f"Loading configuration from {args.config}")

        # Load the configuration
        config = load_yaml(args.config)

        experiment_list = config['experiment_list']
        paths_output_plots = config['output_plots']

        number_dates = config['number_of_dates']
        lead_times = config['Lead_Times']
        var_number = config['var_number']
        names_o_met = config['names_output_metrics']
        names_o_p_folders = config['names_output_plot_folders']
        output_folder = config['output_folder']
        
        size_H = config['size_H']
        size_W = config['size_W']

        assert 'output_plots' in config, f"output_path must be specified in {args.config}"
        if not os.path.exists(config['output_plots']):
            os.mkdir(config['output_plots'])


        ################################# GRAPHS SETUP
        font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 25,
            }
        len_tests = len(experiment_list) # HOW MANY EXPERIMENTS

        N_bins_max = 121 ## IMPORTANT CONSTANT FOR RANK DIAGRAM
        
        ##### SOME VARIABLES FOR REL DIAGRAM #####
        bins = np.linspace(0, 1, num=11)
        freq_obs = np.zeros((10))
        Hit_rate = np.zeros((17))
        false_alarm = np.zeros((17))
        
        ##### SOME VARIABLES FOR ROC
        A_ROC = np.zeros((len_tests))
        A_ROC_skill = np.zeros((len_tests))
        
        Hit_rate[0]=0
        Hit_rate[16]=1
        false_alarm[0]=0
        false_alarm[16]=1
        

        
        ##### ESTHETICS AND TITLE NAMES
        color_p = ['black', 'royalblue', 'darkgreen', 'darkorange', 'red', 'cyan', 'gold', 'pink', 'tan', 'slategray', 'purple', 'palegreen', 'orchid', 'crimson', 'firebrick']


        line = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid','solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid',]


        case_name = [['ff=5 (km/h)', 'ff=10 (km/h)', 'ff=15 (km/h)', 'ff=20 (km/h)', 'ff=30 (km/h)', 'ff=40 (km/h)'],
                    ['', '', '', '', '', ''], 
                    ['t2m=278.15 (K)', 't2m=283.15 (K)', 't2m=288.15 (K)', 't2m=293.15 (K)', 't2m=298.15 (K)', 't2m=303.15 (K)']]
        case_name_thresholds = ['ff' ,'dd', 't2m']
        var_names_m = ['ff (m/s)', 'dd (°)', 't2m (K)'  ]

        cases_clean = config['experiment_names']

        echeance = ['+3H', '', '+9H', '', '+15H', '', '+21H', '', '+27H', '', '+33H', '', '+39H', '', '+45H', '', '+48H', '', '']

        for score_idx in range(len(names_o_p_folders)):
            if not os.path.exists(paths_output_plots + '/' + names_o_p_folders[score_idx] ):
                os.mkdir(paths_output_plots + '/' + names_o_p_folders[score_idx] )

##################################### PLOTTING MEAN BIAS RESULTS ################################
        mean_bias = np.zeros((len_tests, number_dates*lead_times, var_number, size_H, size_W), dtype = ('float32'))
        mean_bias_LT = np.zeros((len_tests, number_dates, lead_times, var_number, size_H, size_W), dtype = ('float32'))

        for i in range(len_tests):
            
            
            mean_bias[i] = np.load(output_folder + '/' + experiment_list[i] + '/' + names_o_met[0] + '.npy')

        D_i = 0
        LT_i = 0
        for i in range(number_dates*lead_times):        
            
            mean_bias_LT[:,D_i, LT_i] = mean_bias[:,i]
            LT_i =LT_i + 1
            
            if LT_i == lead_times : 
                
                D_i = D_i +1
                LT_i = 0
            
        ################################################ MEAN BIAS    
        for i in range(var_number):
            

            fig,axs = plt.subplots(figsize = (9,7))
            
            for k in range(len_tests):
                    
                print(mean_bias_LT[k,:,:,i].shape, )
                
                plt.plot(np.nanmean(mean_bias_LT[k,:,:,i], axis = (0,2,3)), label = cases_clean[k], color= color_p[k], linestyle = line[k] )
                
                axs.set_xticks(range(len(echeance)))
                axs.set_xticklabels(echeance)
                plt.xticks( fontsize ='18')
                axs.tick_params(direction='in', length=12, width=1)

                plt.yticks(fontsize ='18')
                #plt.title(var_names[i] ,fontdict = font)
                #plt.text(0.6, 0.9, var_names[i] + ' ' + domain[ii],
                #        fontdict = font, transform=axs.transAxes)
                plt.ylabel(var_names_m[i], fontsize= '18')
                plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
                plt.savefig(paths_output_plots + '/' + names_o_p_folders[0] + '/' + names_o_p_folders[0] + str(i) + '.pdf')
                plt.close()
        
        mean_bias_LT=0
        mean_bias=0
        gc.collect() 


    ######################## ENSEMBLE CRPS
        crps_scores = np.zeros((len_tests, number_dates*lead_times, var_number), dtype = ('float32'))
        crps_scores_LT = np.zeros((len_tests, number_dates, lead_times, var_number), dtype = ('float32'))
        
        for i in range(len_tests):

            crps = np.load(output_folder + '/' + experiment_list[i] + '/' + names_o_met[1] + '.npy')
            crps_scores[i] = crps[:,:,0]

        D_i = 0
        LT_i = 0
        for i in range(number_dates*lead_times):        
            

            crps_scores_LT[:, D_i, LT_i] = crps_scores[:,i]
            LT_i =LT_i + 1
            if LT_i == lead_times : 
                
                D_i = D_i +1
                LT_i = 0

        for i in range(var_number):
            
            dist_0 = crps_scores_LT[0,:,0:5,i]
            dist_0 = dist_0.reshape(number_dates*5)
            fig,axs = plt.subplots(figsize = (9,7))        
            for k in range(len_tests-1):

                dist = crps_scores_LT[k+1,:,0:5,i]
                dist = dist.reshape(number_dates*5)
                axs.hist(dist-dist_0, bins=50)
                plt.savefig(paths_output_plots + '/' + names_o_p_folders[1] + '/' + names_o_p_folders[1] + '_diff_'+str(i)+'_'+str(k) +'.pdf')
                plt.close()
        
            # We can set the number of bins with the *bins* keyword argument.
            


        for i in range(var_number):
            

                fig,axs = plt.subplots(figsize = (9,7))
                for k in range(len_tests):                      

                    plt.plot(np.nanmean(crps_scores_LT[k,:,:,i], axis=(0)), label=cases_clean[k], color=color_p[k], linestyle = line[k] )

                plt.xticks( fontsize ='18')
                axs.set_xticks(range(len(echeance)))
                axs.set_xticklabels(echeance)
                axs.tick_params(direction='in', length=12, width=1)
                plt.yticks(fontsize ='18')
                plt.ylabel(var_names_m[i], fontsize= '18', fontdict=font)
                #plt.title(var_names[i] + ' ' + domain[ii],fontdict = font)
                #plt.title(var_names[i] ,fontdict = font)                
                #plt.text(0.6, 0.9, var_names[i] + ' ' + domain[ii],
                #        fontdict = font, transform=axs.transAxes)
                plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
                plt.savefig(paths_output_plots + '/' + names_o_p_folders[1] + '/' + names_o_p_folders[1]+str(i) +'.pdf')
                plt.close()
        crps_scores=0
        crps_scores_LT=0
        gc.collect()

    ###########################" SKILLSPREAD

        s_p_scores = np.zeros((len_tests, number_dates*lead_times, 2, var_number, size_H, size_W), dtype = ('float32'))
        s_p_scores_LT = np.zeros((len_tests, number_dates, lead_times, 2, var_number, size_H, size_W), dtype = ('float32'))

        for i in range(len_tests):
            s_p_scores[i] = np.load(output_folder + '/' + experiment_list[i] + '/' + names_o_met[2] + '.npy')

        D_i = 0
        LT_i = 0
        for i in range(number_dates*lead_times):        
            
            s_p_scores_LT[:, D_i, LT_i] = s_p_scores[:,i]
            LT_i =LT_i + 1
            if LT_i == lead_times : 
                D_i = D_i +1
                LT_i = 0
            
            
        for i in range(var_number):
            

                fig,axs = plt.subplots(figsize = (9,7))
                
                for k in range(len_tests):
                        
                    #if k == 0 :    
                        #plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i]**2., axis =(0,2,3))), label = 'SKILL AROME', color= color_p[k], linewidth =3 )
                    
                    plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i]**2., axis =(0,2,3))), label = cases_clean[k], color= color_p[k], linestyle = line[k], linewidth = 2 )
                    plt.plot(np.nanmean(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,1,i], axis =(0))), axis=(-2,-1)), label = cases_clean[k], color= color_p[k], linestyle = line[k] )


                    plt.xticks( fontsize ='18')
                    axs.set_xticks(range(len(echeance)))
                    axs.set_xticklabels(echeance)
                    axs.tick_params(direction='in', length=12, width= 1)
                    plt.yticks(fontsize ='18')
                    plt.ylabel(var_names_m[i], fontsize= '18', fontdict=font)
                    plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
                    plt.savefig(paths_output_plots + '/' + names_o_p_folders[2] + '/' + names_o_p_folders[2]+str(i) +'.pdf')
                plt.close()


        s_p_scores=0
        s_p_scores_LT=0
        gc.collect()


        ####################### BRIER SCORES

        Brier_scores = np.zeros((len_tests, number_dates*lead_times, 6, var_number, size_H, size_W), dtype = ('float32'))
        Brier_scores_LT = np.zeros((len_tests, number_dates, lead_times, 6, var_number, size_H, size_W), dtype = ('float32'))
        
        for i in range(len_tests):

            Brier_scores[i] = np.load(output_folder + '/' + experiment_list[i] + '/' + names_o_met[3] + '.npy')
            
        D_i = 0
        LT_i = 0
        for i in range(number_dates*lead_times):        
            
            Brier_scores_LT[:,D_i, LT_i] = Brier_scores [:, i]
            LT_i =LT_i + 1
    
            if LT_i == lead_times : 
                
                D_i = D_i +1
                LT_i = 0
            
        for i in range(6):
            
            for j in range(var_number):
                fig,axs = plt.subplots(figsize = (9,7))
                for k in range(len_tests):
                                    

                    #plt.plot(np.nanmean(Brier_scores_LT[0,:,:,i, j], axis= (0,2,3))-np.nanmean(Brier_scores_LT[k,:,:,i, j], axis= (0,2,3)), label = cases_clean[k], color = color_p[k], linestyle = line[k])
                    plt.plot(np.nanmean(Brier_scores_LT[k,:,:,i, j], axis= (0,2,3)), label = cases_clean[k], color = color_p[k], linestyle = line[k])
                    
    
                plt.xticks( fontsize ='18')
                axs.tick_params(direction='in', length=12, width=2)
                plt.yticks(fontsize ='18')
                plt.title(case_name[j][i],fontdict = font)
                plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
                plt.savefig(paths_output_plots + '/' + names_o_p_folders[3] + '/' + names_o_p_folders[3]+ '_' + str(i) + '_' + str(j) +'.pdf')
                plt.close()
        for i in range(6):
            
            for j in range(var_number):
                fig,axs = plt.subplots(figsize = (9,7))
                for k in range(len_tests):
                                    

                    #plt.plot(np.nanmean(Brier_scores_LT[0,:,:,i, j], axis= (0,2,3))-np.nanmean(Brier_scores_LT[k,:,:,i, j], axis= (0,2,3)), label = cases_clean[k], color = color_p[k], linestyle = line[k])
                    plt.plot(np.nanmean(Brier_scores_LT[0,:,:,i, j] - Brier_scores_LT[k,:,:,i, j], axis= (0,2,3)), label = cases_clean[k], color = color_p[k], linestyle = line[k])
                    
    
                plt.xticks( fontsize ='18')
                axs.tick_params(direction='in', length=12, width=2)
                plt.yticks(fontsize ='18')
                plt.title(case_name[j][i],fontdict = font)
                plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
                plt.savefig(paths_output_plots + '/' + names_o_p_folders[3] + '/' + names_o_p_folders[3]+ '_diff_' + str(i) + '_' + str(j) +'.pdf')
                plt.close()
        
        for j in range(var_number):
            for k in range(len_tests):
                brier_diff = np.zeros((6,))
                for i in range(6):
                    brier_diff[i] = np.nanmean(Brier_scores_LT[0,:,:,i, j] - Brier_scores_LT[k,:,:,i, j])
                plt.plot(brier_diff, label = cases_clean[k], color = color_p[k], linestyle = line[k])
            plt.xticks( fontsize ='18')
            axs.set_xticks(range(len(case_name[j])))
            axs.set_xticklabels(case_name[j])
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name_thresholds[j],fontdict = font)
            plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
            plt.savefig(paths_output_plots + '/' + names_o_p_folders[3] + '/' + names_o_p_folders[3]+ '_diff_thresholds_' + str(j) + '.pdf')
            plt.close()

        Brier_scores=0
        Brier_scores_LT=0
        gc.collect()

        #### RANK HISTOGRAM
        if 'rankHistogram' in names_o_p_folders:
            rank_histo = np.zeros((len_tests, number_dates*lead_times, var_number, N_bins_max))


            for i in range(len_tests):

                rank_histo[i] = np.load(output_folder + '/' + experiment_list[i] + '/' + names_o_met[4] + '.npy')


            N_bins= [config['experiments'][0]['dataloaders']['N_ens'] + 1 for k in range(len_tests)] ######## THIS IS STILL HARD CODED...
            

            for j in range(var_number):
                for k in range(len_tests):
                    fig,axs = plt.subplots(figsize = (9,7))
                    ind = np.arange(N_bins[k])
                    print(rank_histo[k,:,j,0:N_bins[k]].sum(axis=0).shape)
                    plt.bar(ind, rank_histo[k,:,j,0:N_bins[k]].sum(axis=0))
                    plt.title(cases_clean[k] + ' ' + var_names_m[j],fontdict = font)
                    #plt.xticks( fontsize ='18')
                    plt.tick_params(bottom = False, labelbottom = False)
                    plt.xlabel('Bins', fontsize= '18')
                    plt.ylabel('Number of Observations', fontsize= '18')
                    axs.tick_params(length=12, width=1)
                    plt.yticks(fontsize ='18')

                    plt.savefig(paths_output_plots + '/' + names_o_p_folders[4] + '/' + names_o_p_folders[4] + '_' + str(j) + '_' + str(k) +'.pdf')

            rank_histo=0
            gc.collect()

#####################################################"PLOT REL DIAGRAM
        

        rel_diag_scores = np.zeros((len_tests,number_dates*lead_times, 6, 2, var_number, size_H, size_W))
        for i in range(len_tests):
            
            rel_diag_scores[i] = np.load(output_folder + '/' + experiment_list[i] + '/' + names_o_met[5] + '.npy')



        for i in range(6):
            
            for j in range(var_number):
                fig,axs = plt.subplots(figsize = (9,7))
                for k in range(len_tests):
                    O_tr = rel_diag_scores[k,:-2,i,1,j]
                    X_prob = rel_diag_scores[k,:-2,i,0,j]
                    
                    for z in range(bins.shape[0]-1):
                        
                        obs = copy.deepcopy(O_tr[np.where((X_prob >= bins[z]) & (X_prob < bins[z+1]), True, False)])
                        obs = obs[~np.isnan(obs)]
                        print(obs.shape, j)
                        freq_obs[z] = obs.sum()/obs.shape[0]
                    plt.plot(bins[:-1]+0.05, freq_obs, label = cases_clean[k], color = color_p[k], linestyle = line[k])
                        
                plt.plot(bins[:-1]+0.05, bins[:-1]+0.05, label = 'perfect', color = 'black', linewidth =3 ) ### I don't remember why I'm adding 0.05
                plt.xticks( fontsize ='18')
                plt.xlabel('forecast probability', fontsize= '18')
                plt.ylabel('observation frequency', fontsize= '18')
                axs.tick_params(direction='in', length=12, width=1)
                plt.yticks(fontsize ='18')
                plt.title(case_name[j][i],fontdict = font)
                plt.legend(fontsize = 10, ncol=1, frameon = False, loc='lower right')
                plt.savefig(paths_output_plots + '/' + names_o_p_folders[5] + '/' + names_o_p_folders[5]+'_' + str(i) + '_' + str(j) +'.pdf')
                
        
        rel_diag_scores=0
        gc.collect()

######################################"" ROC

        rel_diag_scores = np.zeros((len_tests,number_dates*lead_times, 6, 2, var_number, size_H, size_W))
        for i in range(len_tests):
            
            rel_diag_scores[i] = np.load(output_folder + '/' + experiment_list[i] + '/' + names_o_met[5] + '.npy') ### ATENTION ROC USES SCORES FROM REL_DIAG_SCORES



        bins_roc = np.array([0.99, 0.93, 0.86, 0.79, 0.72, 0.65, 0.58, 0.51, 0.44, 0.37, 0.3, 0.23, 0.14, 0.07, 0.01])
        for i in range(6):

            for j in range(var_number):
                fig,axs = plt.subplots(figsize = (9,7))
                for k in range(len_tests):
                    O_tr = rel_diag_scores[k,:-2,i,1,j]
                    X_prob = rel_diag_scores[k,:-2,i,0,j]

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
                        


                    plt.plot(false_alarm, Hit_rate, label = cases_clean[k], color = color_p[k], linestyle = line[k])
    
                    A_ROC[k] = np.trapz(Hit_rate, false_alarm)
                        
                    A_ROC_skill[k]=1-A_ROC[0]/A_ROC[k]
                    


                plt.xticks( fontsize ='18')
                plt.xlabel('False Alarm Rate', fontsize= '18')
                plt.ylabel('Hit Rate', fontsize= '18')
                axs.tick_params(direction='in', length=12, width=2)
                plt.yticks(fontsize ='18')
                plt.title(case_name[j][i],fontdict = font)
                plt.legend(fontsize = 14,frameon = False, ncol=1)
                


                plt.savefig(paths_output_plots + '/' + names_o_p_folders[6] + '/' + names_o_p_folders[6]+'_' + str(i) + '_' + str(j) +'.pdf')
                
                fig,axs = plt.subplots(figsize = (9,7))
                
                fig,axs = plt.subplots(figsize = (9,7))
                plt.bar(cases_clean[1::], A_ROC_skill[1::])
                plt.xticks( fontsize ='18')
                plt.ylabel('Area under ROC skill', fontsize= '18')
                axs.tick_params(direction='in', length=12, width=2)
                plt.yticks(fontsize ='18')
                plt.title(case_name[j][i],fontdict = font)
                
                plt.savefig(paths_output_plots + '/' + names_o_p_folders[6] + '/' + names_o_p_folders[6]+ '_' + 'AROC'+'_' + str(i) + '_' + str(j) +'.pdf')



        logger.info("Program completed.")
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
