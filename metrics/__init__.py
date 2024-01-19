import metrics.general_metrics
import metrics.length_scales as ls
import metrics.spectrum_analysis as spec
import metrics.sliced_wasserstein as SWD
import metrics.wasserstein_distances as WD
import metrics.quantiles_metric as quant
import metrics.multivariate as multiv
import metrics.brier_score as BS
# import metrics.CRPS_calc as CRPS_calc
import metrics.skill_spread as SP
import metrics.spectral_variance as spvar
import metrics.rank_histogram as RH
import metrics.bias_ensemble as BE
import metrics.mean_bias as mb
import metrics.skill_spread_deviation as skspd
from metrics import CRPS_calc

from metrics.metrics import Metric

class W1CenterNUMPY(Metric):
    def __init__(self, name):
        super().__init__(isBatched=False, names=['W1_Center'])

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return WD.W1_center_numpy(real_data,fake_data)

class W1RandomNUMPY(Metric):
    def __init__(self, name):
        super().__init__(isBatched=False, names=['W1_random'], var_channel=1)

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        return WD.W1_random_NUMPY(real_data,fake_data)

class pwW1(Metric):
    def __init__(self, name):
        super().__init__(isBatched=False, names=['pw_W1'])

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return WD.pointwise_W1(real_data,fake_data)


#######################################################################
######################### Sliced Wasserstein Distance estimations #####
#######################################################################


class SWDall(Metric):
    def __init__(self, image_shape=(256,256)):
        super().__init__(isBatched=False)
        self.sliced_w1 = SWD.SWD_API(image_shape=image_shape, numpy=True)
        self.names = sliced_w1.get_metric_names()


    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return self.sliced_w1.End2End(real_data,fake_data)

class SWDallTorch(Metric):
    def __init__(self, name, image_shape=(256,256)):
        super().__init__(isBatched=False)
        self.sliced_w1_torch = SWD.SWD_API(image_shape=image_shape, numpy=False)
        self.names = self.sliced_w1_torch.get_metric_names()

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        
        return self.sliced_w1_torch.End2End(real_data,fake_data)

#######################################################################
######################### Spectral Analysis Metrics ###################
#######################################################################

class spectralCompute(Metric):
    """
    DCT computation for Power Spectral Density
    """
    def __init__(self, name):
        super().__init__(isBatched=False)

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_standalone(fake_data)

    def _calculateCore(self, processed_data):
        return spec.PowerSpectralDensity(processed_data)

class spectralDist(Metric):
    """
    DCT computation for Power Spectral Density
    """
    def __init__(self, name):
        super().__init__(isBatched=False)

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return spec.PSD_compare(real_data,fake_data)

class spectralDistMultidates(Metric):
    def __init__(self, name):
        super().__init__(isBatched=False)

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return spec.PSD_compare_multidates(real_data,fake_data)

###################################################################
######################### Length Scales Metrics ###################
###################################################################

class lsMetric(Metric):
    """
    Correlation length maps
    """
    def __init__(self, scale=2.5):
        super().__init__(isBatched=False)
        self.scale = scale

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_standalone(fake_data)

    def _calculateCore(self, processed_data):
        return ls.length_scales(processed_data, self.scale)


class lsDist(Metric):
    """
    MAE on correlation length maps
    """
    def __init__(self, scale=2.5):
        super().__init__(isBatched=False, names = ['Lcorr_u', 'Lcorr_v', 'Lcorr_t2m'])
        self.scale = scale

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)
            
    def _calculateCore(self, processed_data):

        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return ls.length_scale_abs(real_data, fake_data, self.scale)

###################################################################
######################### Quantiles Metric ########################
###################################################################

qlist = [0.01,0.1,0.9,0.99]
class Quantiles(Metric):
    def __init__(self, qlist = [0.01,0.1,0.9,0.99]):
        super().__init__(isBatched=False)
        self.qlist = qlist

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_standalone(fake_data)

    
    def _calculateCore(self, processed_data):
        return quant.quantiles(processed_data, self.qlist)

class QuantilesScore(Metric):
    def __init__(self, qlist = [0.01,0.1,0.9,0.99]):
        super().__init__(isBatched=False)
        self.qlist = qlist

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)
    
    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        return quant.quantile_score(real_data, fake_data, self.qlist)

#####################################################################
######################### Multivariate Correlations #################
#####################################################################

class MultivarCorr(Metric):
    def __init__(self, names=['Corr_r','Corr_f']):
        super().__init__(isBatched=False)

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)
    
    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return multiv.multi_variate_correlations(real_data, fake_data)

#######################################################################
#######################################################################
#######################################################################
 
######################### Ensemble Metrics ############################

#######################################################################
#######################################################################
#######################################################################



#####################################################################
############################ CRPS ###################################
#####################################################################
#
class ensembleCRPS(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['CRPSff', 'CRPSdd','CRPSt2m'])
        self.debiasing = False

    def _preprocess(self, fake_data, real_data=None, obs_data=None, debiasing=None, debiasing_mode=None, conditioning_members=None):
        return self.preprocess_cond_obs(fake_data, real_data, obs_data, debiasing, debiasing_mode, conditioning_members)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']
        return CRPS_calc.ensemble_crps(obs_data,real_data,fake_data)

class crpsMultiDates(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['CRPSff', 'CRPSdd','CRPSt2m'])
        self.debiasing = False

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_cond_obs(fake_data, real_data, obs_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']

        return CRPS_calc.crps_multi_dates(obs_data,real_data,fake_data, debiasing=self.debiasing)

class crpsDiffMultiDates(Metric):
    def __init__(self, name):
        super().__init__(isBatched=False, names=['CRPSff', 'CRPSdd','CRPSt2m'], debiasing=False)
        self.debiasing = debiasing

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_cond_obs(fake_data, real_data, obs_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']

        return CRPS_calc.crps_vs_aro_multi_dates(obs_data,real_data,fake_data, debiasing=self.debiasing)

#####################################################################
############################ Brier ##################################
#####################################################################

class brierScore(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['Brierff', 'Brierdd','Briert2m'])
        self.debiasing = False

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_cond_obs(fake_data, real_data, obs_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']

        return BS.brier_score(obs_data,real_data,fake_data, debiasing=self.debiasing)

#####################################################################
############################ Skill-Spread ###########################
#####################################################################

class skillSpread(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['Skspff', 'Skspdd','Skspt2m'])
        self.debiasing = False

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_cond_obs(fake_data, real_data, obs_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']

        return SP.skill_spread(obs_data,real_data,fake_data, debiasing=self.debiasing)

class skillSpreadDeviationMultidates(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['Skspff', 'Skspdd','Skspt2m'], debiasing=False)
        self.debiasing = debiasing

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_cond_obs(fake_data, real_data, obs_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']

        return skspd.skill_spread_deviation_multidates(obs_data,real_data,fake_data, debiasing=self.debiasing)

class thresholdedSkillSpreadDeviationMultidates(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['Skspff', 'Skspdd','Skspt2m'], debiasing=False)
        self.debiasing = debiasing

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_cond_obs(fake_data, real_data, obs_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']

        return skspd.thresholded_skill_spread_deviation_multidates(obs_data,real_data,fake_data, debiasing=self.debiasing)



#####################################################################
############################ Rank Histogram #########################
#####################################################################

class rankHistogram(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['Skspff', 'Skspdd','Skspt2m'], debiasing=False)
        self.debiasing = debiasing

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_cond_obs(fake_data, real_data, obs_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']

        return RH.rank_histo(obs_data,real_data,fake_data)


#####################################################################
############################ Reliability ############################
#####################################################################

class relDiagram(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['Relff', 'Reldd','Relt2m'], debiasing=False)
        self.debiasing = debiasing

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_cond_obs(fake_data, real_data, obs_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']

        return RD.rel_diag(obs_data,real_data,fake_data)

#####################################################################
############################ Bias ###################################
#####################################################################

class biasEnsemble(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['Biasff', 'Biasdd','Biast2m'])

    def _preprocess(self, fake_data, real_data=None, obs_data=None, debiasing=None):
        
        return self.preprocess_cond_obs(fake_data, real_data, obs_data, debiasing)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']

        return BE.bias_ens(obs_data,fake_data,real_data)

class meanBias(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['Biasff', 'Biasdd','Biast2m'])
        self.debiasing = debiasing

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_cond_obs(fake_data, real_data, obs_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']

        return mb.mean_bias(obs_data,real_data,fake_data)


#####################################################################
############################ Spread only ############################
#####################################################################

class variance(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['varu', 'varv','vart2m'])

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_standalone(fake_data)

    def _calculateCore(self, processed_data):
        fake_data = processed_data['fake_data']

        return BE.bias_ens(obs_data,real_data,fake_data)

class varianceDiff(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['var_diff_u', 'var_diff_v','var_diff_t2m'])

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return GM.variance_diff(real_data,fake_data)

class stdDiff(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['std_diff_u', 'std_diff_v','std_diff_t2m'])

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return GM.std_diff(real_data,fake_data)


class relStdDiff(Metric):
    def __init__(self, name):
        super().__init__(isBatched=True, names=['rel_std_diff_u', 'rel_std_diff_v','rel_std_diff_t2m'])

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        return self.preprocess_dist(real_data,fake_data)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return GM.relative_std_diff(real_data,fake_data)

#####################################################################
############################ Spectral analysis ######################
#####################################################################