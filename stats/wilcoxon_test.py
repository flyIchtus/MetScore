import numpy as np
from scipy.stats import wilcoxon

def group_by_leadtime(scores,scores_LT,config):
    D_i = 0
    LT_i = 0
    print("scores shape", scores.shape)
    for timestamp in range(config['number_dates'] * config['lead_times']):        
        scores_LT[:,D_i, LT_i] = scores[:,timestamp]
        LT_i = LT_i + 1
        if LT_i==config['lead_times']: 
            D_i = D_i +1
            LT_i = 0
    print("scores LT shape", scores_LT.shape)
    return scores_LT


def computeStats(experiments, scores_LT, config):

    decisions = [[] for var_idx in range(config['var_number'])]
    results = [[] for var_idx in range(config['var_number'])]
    for var_idx in range(config['var_number']):
        stats_decision = []
        results_var = []
        ### reference experiment is experiment 0
        for exp_idx, exp in enumerate(experiments[1:]):
            print(exp['short_name'], var_idx)
            diff = scores_LT[exp_idx + 1,:,:,var_idx] - scores_LT[0,:,:,var_idx]
            res_diff = wilcoxon(diff,axis=0,zero_method='zsplit')
            print("res_diff shape", res_diff.statistic)
            results_var.append(res_diff.statistic[np.newaxis,:])
            # rejecting the null hypothesis <=> distributions are different
            decision_different = np.where((res_diff.pvalue<=0.05), True, False)
            print("decision different",decision_different)
            res_greater = wilcoxon(diff, axis=0, alternative='greater',zero_method='zsplit')
            decision_greater = np.where((res_greater.pvalue<=0.05), True, False)
            print("decision greater",decision_greater)
            res_less = wilcoxon(diff, axis=0, alternative='less',zero_method='zsplit')
            decision_less = np.where((res_less.pvalue<=0.05), True, False)
            print("decision less",decision_less)
            stats_decision.append((decision_different))
            #(decision_different + 2 * decision_greater + 3 * decision_less)
        
        decisions[var_idx].append(np.array(stats_decision))
        print("decision aggreg exp", decisions[var_idx][-1].shape)
        results[var_idx].append(np.array(results_var))

    return np.array(decisions), np.array(results)
        
def decision_leadtimes(scores):
    print(scores.shape)
    res = wilcoxon(scores,zero_method='zsplit')
    return (res.pvalue<=0.05)

def load_and_format_scores(experiments, metric, config):
    exp = experiments[0]
    print(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')
    data_model = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy')
    Shape = data_model.shape
    print(f"Scores {metric['name']} loaded, model {Shape}")
    if len(Shape)==4:
        print('no split')
        scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], config['var_number']))
        scores_LT = np.zeros((len(experiments), config['number_dates'], config['lead_times'], config['var_number']))

        scores[0] = np.nanmean(data_model,axis=(-2,-1))
        for exp_idx, exp in enumerate(experiments[1:]):
            scores[exp_idx+1] = np.nanmean(np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy'),axis=(-2,-1))
        scores_LT = group_by_leadtime(scores, scores_LT,config)
        return [scores_LT]
    elif len(Shape)==5:
        print('split involved')
        scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], Shape[1], config['var_number']))
        scores_LT = np.zeros((len(experiments), config['number_dates'], config['lead_times'], Shape[1],config['var_number']))

        scores[0] = np.nanmean(data_model,axis=(-2,-1))
        for exp_idx, exp in enumerate(experiments[1:]):
            scores[exp_idx+1] = np.nanmean(np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy'),axis=(-2,-1))
        scores_LT = np.split(group_by_leadtime(scores, scores_LT,config),Shape[1],axis=3)
        print(len(scores_LT), scores_LT[0].shape)
        return scores_LT
    elif len(Shape)==3:
        print('no split, 3')
        scores = np.zeros((len(experiments), config['number_dates'] * config['lead_times'], config['var_number']))
        scores_LT = np.zeros((len(experiments), config['number_dates'], config['lead_times'], config['var_number']))

        scores[0] = (data_model).squeeze()
        for exp_idx, exp in enumerate(experiments[1:]):
            scores[exp_idx+1] = np.load(config['expe_folder'] + '/' + exp['name'] + '/' + metric['name'] + '.npy').squeeze()
        scores_LT = group_by_leadtime(scores, scores_LT,config)
        return [scores_LT]

def significance(experiments, metric, config):

    scores_list = load_and_format_scores(experiments, metric, config)
    print(len(scores_list))
    for score_idx, scores_LT in enumerate(scores_list):
        decisions, results = computeStats(experiments, scores_LT.squeeze(), config)
        print(decisions.shape, results.shape)
        
        np.save(f"{config['output_plots']}/{metric['folder']}/{metric['name']}_decisions_{score_idx}.npy",decisions)
        np.save(f"{config['output_plots']}/{metric['folder']}/{metric['name']}_statistics_{score_idx}.npy",results)