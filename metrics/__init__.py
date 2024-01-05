import metrics.general_metrics
import metrics.length_scales as ls
import metrics.spectrum_analysis as spec
import metrics.sliced_wasserstein as SWD
import metrics.wasserstein_distances as WD
import metrics.quantiles_metric as quant
import metrics.multivariate as multiv

from metrics.metrics import Metric

#######################################################################
######################### PointWise Wasserstein Distance estimations ##
#######################################################################

class W1CenterNUMPY(Metric):
    def __init__(self):
        super().__init__(isBatched=False, names=['W1_Center'])

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables
    
        if len(self.var_indices)==fake_data.shape[self.var_channel]:
            return {'real_data': real_data,
                    'fake_data': fake_data}
        else:       
            return {'real_data': real_data.take(indices=self.real_var_indices, axis=self.var_channel),
                    'fake_data': fake_data.take(indices=self.var_indices, axis=self.var_channel)}

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return WD.W1_center_numpy(real_data,fake_data)

class W1RandomNUMPY(Metric):
    def __init__(self):
        super().__init__(isBatched=False, names=['W1_random'])

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables
    
        if len(self.var_indices)==fake_data.shape[self.var_channel]:
            return {'real_data': real_data,
                    'fake_data': fake_data}
        else:       
            return {'real_data': real_data.take(indices=self.real_var_indices, axis=self.var_channel),
                    'fake_data': fake_data.take(indices=self.var_indices, axis=self.var_channel)}

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return WD.W1_random_NUMPY(real_data,fake_data)

class pwW1(Metric):
    def __init__(self):
        super().__init__(isBatched=False, names=['pw_W1'])

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables
    
        if len(self.var_indices)==fake_data.shape[self.var_channel]:
            return {'real_data': real_data,
                    'fake_data': fake_data}
        else:       
            return {'real_data': real_data.take(indices=self.real_var_indices, axis=self.var_channel),
                    'fake_data': fake_data.take(indices=self.var_indices, axis=self.var_channel)}

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
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables
    
        if len(self.var_indices)==fake_data.shape[self.var_channel]:
            return {'real_data': real_data,
                    'fake_data': fake_data}
        else:       
            return {'real_data': real_data.take(indices=self.real_var_indices, axis=self.var_channel),
                    'fake_data': fake_data.take(indices=self.var_indices, axis=self.var_channel)}

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return self.sliced_w1.End2End(real_data,fake_data)

class SWDallTorch(Metric):
    def __init__(self, image_shape=(256,256)):
        super().__init__(isBatched=False)
        self.sliced_w1_torch = SWD.SWD_API(image_shape=image_shape, numpy=False)
        self.names = self.sliced_w1_torch.get_metric_names()

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables        
        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            return {'real_data': real_data,
                    'fake_data': fake_data}
        else:       
            return {'real_data': real_data.take(indices=self.real_var_indices, axis=self.var_channel),
                    'fake_data': fake_data.take(indices=self.var_indices, axis=self.var_channel)}

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
    def __init__(self):
        super().__init__(isBatched=False)

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables   
        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data

        return fake_data_p

    def _calculateCore(self, processed_data):
        return spec.PowerSpectralDensity(processed_data)

class spectralDist(Metric):
    """
    DCT computation for Power Spectral Density
    """
    def __init__(self):
        super().__init__(isBatched=False)

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables        
        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data

        if len(self.real_var_indices)!=real_data.shape[self.var_channel]:
            real_data_p = real_data.take(indices=self.real_var_indices, axis=self.var_channel)
        else:
            real_data_p = real_data
        
        return {'real_data': real_data_p,
                'fake_data': fake_data_p}

    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return spec.PSD_compare(real_data, fake_data)

class spectralDistMultidates(Metric):
    def __init__(self):
        super().__init__(isBatched=False)

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables

        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data

        if len(self.real_var_indices)!=real_data.shape[self.var_channel]:
            real_data_p = real_data.take(indices=self.real_var_indices, axis=self.var_channel)
        else:
            real_data_p = real_data
        
        return {'real_data': real_data_p,
                'fake_data': fake_data_p}

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
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables
        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data

        return fake_data_p

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
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables
        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data

        if len(self.real_var_indices)!=real_data.shape[self.var_channel]:
            real_data_p = real_data.take(indices=self.real_var_indices, axis=self.var_channel)
        else:
            real_data_p = real_data
        
        return {'real_data': real_data_p,
                'fake_data': fake_data_p}
            
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
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables
        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data

        return fake_data_p
    
    def _calculateCore(self, processed_data):
        return quant.quantiles(processed_data, self.qlist)

class QuantilesScore(Metric):
    def __init__(self, qlist = [0.01,0.1,0.9,0.99]):
        super().__init__(isBatched=False)
        self.qlist = qlist

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables
        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data

        if len(self.real_var_indices)!=real_data.shape[self.var_channel]:
            real_data_p = real_data.take(indices=self.real_var_indices, axis=self.var_channel)
        else:
            real_data_p = real_data
        
        return {'real_data': real_data_p,
                'fake_data': fake_data_p}
    
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
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables
        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data

        if len(self.real_var_indices)!=real_data.shape[self.var_channel]:
            real_data_p = real_data.take(indices=self.real_var_indices, axis=self.var_channel)
        else:
            real_data_p = real_data

        return {'real_data': real_data_p,
                'fake_data': fake_data_p}
    
    def _calculateCore(self, processed_data):
        real_data = processed_data['real_data']
        fake_data = processed_data['fake_data']

        return multiv.multi_variate_correlations(real_data, fake_data)