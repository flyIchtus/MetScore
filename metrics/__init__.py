import numpy as np

import metrics.CRPS_calc as CRPS_calc
import metrics.bias_ensemble as BE
import metrics.brier_score as BS
import metrics.general_metrics as GM
import metrics.length_scales as ls
import metrics.mean_bias as mb
import metrics.multivariate as multiv
import metrics.quantiles_metric as quant
import metrics.rank_histogram as RH
import metrics.rel_diagram as RD
import metrics.skill_spread as SP
import metrics.skill_spread_deviation as skspd
import metrics.sliced_wasserstein as SWD
import metrics.spectral_variance as spvar
import metrics.spectrum_analysis as spec
import metrics.wasserstein_distances as WD
from metrics import CRPS_calc
from metrics import area_proportion as ap
from metrics import object_detection as obj

from metrics.metrics import Metric, PreprocessCondObs, PreprocessDist, PreprocessStandalone


class W1CenterNUMPY(PreprocessDist):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        return WD.W1_center_numpy(real_data,fake_data)

class W1RandomNUMPY(PreprocessDist):
    def __init__(self, *args , **kwargs):
        super().__init__(isBatched=False)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return WD.W1_random_NUMPY(real_data,fake_data)

class pwW1(PreprocessDist):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return WD.pointwise_W1(real_data,fake_data)


#######################################################################
######################### Sliced Wasserstein Distance estimations #####
#######################################################################


class SWDall(PreprocessDist):
    def __init__(self, *args, image_shape=(256,256) , **kwargs):
        super().__init__(isBatched=False, **kwargs)
        self.sliced_w1 = SWD.SWD_API(image_shape=image_shape, numpy=True)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return self.sliced_w1.End2End(real_data,fake_data)

class SWDallTorch(PreprocessDist):
    def __init__(self, *args, image_shape=(256,256) , **kwargs):
        super().__init__(isBatched=False, **kwargs)
        self.sliced_w1_torch = SWD.SWD_API(image_shape=image_shape, numpy=False)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        
        return self.sliced_w1_torch.End2End(real_data,fake_data)

#######################################################################
######################### Spectral Analysis Metrics ###################
#######################################################################

class spectralCompute(PreprocessStandalone):
    """
    DCT computation for Power Spectral Density
    """
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False, **kwargs)

    def _calculateCore(self, processed_data):
        return spec.PowerSpectralDensity(processed_data)

class spectralDist(PreprocessDist):
    """
    DCT computation for Power Spectral Density
    """
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False, **kwargs)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return spec.PSD_compare(real_data,fake_data)

class spectralDistMultidates(PreprocessDist):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False, **kwargs)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return spec.PSD_compare_multidates(real_data,fake_data)

#######################################################################
######################### Precipitation physics metrics  ##############
#######################################################################

class AreaProportion(PreprocessStandalone):
    """
    DCT computation for Power Spectral Density
    """
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False, **kwargs)

    def _calculateCore(self, processed_data):
        return ap.area_greater_than(processed_data, self.rr_idx)

class QuantilesThresholded(PreprocessStandalone):
    """
    DCT computation for Power Spectral Density
    """
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False, **kwargs)

    def _calculateCore(self, processed_data):
        return quant.quantiles_non_zero(processed_data, self.qlist)

class ObjectsAttribution(PreprocessStandalone):
    def __init__(self,*args,**kwargs):
        super().__init__(isBatched=False,**kwargs)
        self.zone = obj.Zone(X_min=args[0]['lon_min'],Y_min=args[0]['lat_min'],nb_lon=args[0]['sizeW'],nb_lat=args[0]['sizeH'])

    def _calculateCore(self, processed_data):
        return obj.batchAttributes(processed_data,self.zone,self.rr_idx)


###################################################################
######################### Length Scales Metrics ###################
###################################################################

class lsMetric(PreprocessStandalone):
    """
    Correlation length maps
    """

    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False, **kwargs)

    def _calculateCore(self, processed_data):
        return ls.length_scale(processed_data, self.scale)


class lsDist(PreprocessDist):
    """
    MAE on correlation length maps
    """
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False, names = ['Lcorr_u', 'Lcorr_v', 'Lcorr_t2m'])
            
    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        return ls.length_scale_abs(real_data, fake_data, self.scale)

###################################################################
######################### Quantiles Metric ########################
###################################################################

# qlist = [0.01,0.1,0.9,0.99]
class Quantiles(PreprocessStandalone):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False, **kwargs)
    
    def _calculateCore(self, processed_data):
        return quant.quantiles(processed_data, self.qlist)

class QuantilesScore(PreprocessDist):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False, **kwargs)
    
    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        return quant.quantile_score(real_data, fake_data, self.qlist)

#####################################################################
######################### Multivariate Correlations #################
#####################################################################

class MultivarCorr(PreprocessDist):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False, **kwargs)
    
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
class ensembleCRPS(PreprocessCondObs):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)

    def _calculateCore(self, processed_data):
        if not self.isOnReal:
            exp_data = processed_data['fake_data']
        else:
            exp_data = processed_data['real_data']
        obs_data = processed_data['obs_data']
        return CRPS_calc.ensemble_crps(obs_data, exp_data, self.fair)

class crpsMultiDates(PreprocessCondObs):

    required_keys = ['debiasing','isOnReal']

    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']
        if self.isOnReal:
            return CRPS_calc.crps_multi_dates(obs_data,real_data,real_data, debiasing=self.debiasing)
        else:
            return CRPS_calc.crps_multi_dates(obs_data,real_data,fake_data, debiasing=self.debiasing)

class crpsDiffMultiDates(PreprocessCondObs):

    required_keys = ['debiasing','isOnReal']

    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=False)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']
        if self.isOnReal:
            return CRPS_calc.crps_vs_aro_multi_dates(obs_data,real_data,real_data, debiasing=self.debiasing)
        else:
            return CRPS_calc.crps_vs_aro_multi_dates(obs_data,real_data,fake_data, debiasing=self.debiasing)

        

#####################################################################
############################ Brier ##################################
#####################################################################

class brierScore(PreprocessCondObs):

    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)


    def _calculateCore(self, processed_data):
        if not self.isOnReal:
            exp_data = processed_data['fake_data']
        else:
            exp_data = processed_data['real_data']
        obs_data = processed_data['obs_data']

        return BS.brier_score(obs_data,exp_data,np.array(self.threshold))


class brierSkillScore(PreprocessCondObs):

    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)


    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']

        return BS.brier_skill_score(obs_data,real_data,fake_data,np.array(self.threshold))
#####################################################################
############################ Skill-Spread ###########################
#####################################################################

class skillSpread(PreprocessCondObs):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)

    def _calculateCore(self, processed_data):
        if not self.isOnReal:
            exp_data = processed_data['fake_data']
        else:
            exp_data = processed_data['real_data']
        obs_data = processed_data['obs_data']
        return SP.skill_spread(obs_data,exp_data)

class skillSpreadDeviationMultidates(PreprocessCondObs):

    required_keys = ['debiasing','isOnReal']

    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']
        if self.isOnReal:
            return skspd.skill_spread_deviation_multidates(obs_data,real_data,real_data, debiasing=self.debiasing)
        else:
            return skspd.skill_spread_deviation_multidates(obs_data,real_data,fake_data, debiasing=self.debiasing)


class thresholdedSkillSpreadDeviationMultidates(PreprocessCondObs):

    required_keys = ['debiasing','isOnReal']

    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']
        if self.isOnReal:
            return skspd.thresholded_skill_spread_deviation_multidates(obs_data,real_data,real_data, debiasing=self.debiasing)
        else:
            return skspd.thresholded_skill_spread_deviation_multidates(obs_data,real_data,fake_data, debiasing=self.debiasing)


#####################################################################
############################ Rank Histogram #########################
#####################################################################

class rankHistogram(PreprocessCondObs):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)

    def _calculateCore(self, processed_data):
        if not self.isOnReal:
            exp_data = processed_data['fake_data']
        else:
            exp_data = processed_data['real_data']
        obs_data = processed_data['obs_data']

        return RH.rank_histo(obs_data,exp_data)

#####################################################################
############################ Reliability ############################
#####################################################################

class relDiagram(PreprocessCondObs):

    required_keys = ['threshold']

    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)

    def _calculateCore(self, processed_data):
        if not self.isOnReal:
            exp_data = processed_data['fake_data']
        else:
            exp_data = processed_data['real_data']
        obs_data = processed_data['obs_data']

        return RD.rel_diag(obs_data, exp_data, np.array(self.threshold))

#####################################################################
############################ Bias ###################################
#####################################################################

class biasEnsemble(PreprocessCondObs):
    required_keys = ['threshold', 'isOnReal']

    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)


    def _calculateCore(self, processed_data):
        if not self.isOnReal:
            exp_data = processed_data['fake_data']
        else:
            exp_data = processed_data['real_data']
        obs_data = processed_data['obs_data']

        return BE.bias_ens(obs_data,exp_data)

class meanBias(PreprocessCondObs):

    required_keys = ['isOnReal']

    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']
        obs_data = processed_data['obs_data']
        if self.isOnReal:
            return mb.mean_bias(obs_data,real_data,fake_data)
        else:
            return mb.mean_bias(obs_data,real_data,fake_data)


#####################################################################
############################ Spread only ############################
#####################################################################

class variance(PreprocessStandalone):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)

    def _calculateCore(self, processed_data):
        return GM.simple_variance(processed_data)

class varianceDiff(PreprocessDist):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return GM.variance_diff(real_data,fake_data)

class stdDiff(PreprocessDist):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)


    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return GM.std_diff(real_data,fake_data)


class relStdDiff(PreprocessDist):
    def __init__(self, *args, **kwargs):
        super().__init__(isBatched=True)

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return GM.relative_std_diff(real_data,fake_data)

#####################################################################
        ######################################################
#####################################################################