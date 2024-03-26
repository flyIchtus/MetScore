import seaborn as sns
import numpy as np
import logging
from itertools import combinations_with_replacement
import pandas as pd

def space2batchLongForm(data, offset):
    """
    Transform batched image data as b x C x H x W to (bHW) x C
    offset is if data is cropped by the sides
        b x C x H x W --> (b(H-2*offset)(W-2*offset)) x C
    """

    Shape = data.shape
    assert len(Shape)==4
    
    data_list = []
    for i in range(Shape[1]):
        if offset>0:
            data_list.append(np.expand_dims(data[:,i,offset:-offset,offset:-offset].reshape(Shape[0] * (Shape[2] - 2 * offset) * (Shape[3] - 2 * offset)), axis= 1))
        else:
            data_list.append(np.expand_dims(data[:,i].reshape(Shape[0] * (Shape[2]) * (Shape[3])), axis= 1))
    
    a = np.concatenate(data_list, axis = 1).T
    return a


def multi_variate_correlations(data_real, data_fake, variables = ['u','v','t2m'], offset=0):
    """
    To be used in the metrics evaluation framework
    data_r, data_f : numpy arrays, shape B xC x H xW
    
    Returns :
        
        Out_rf : numpy array, shape 2 x C*(C-1)//2 x nbins 
          bivariates histograms for [0,:,:] -> real samples
                                    [1,:,:] -> fake samples
    
    """
    
    channels = data_fake.shape[1]
    couples = combinations_with_replacement(range(channels), 2)
    assert channels==len(variables)
    ncouples2 = channels * (channels-1) // 2
    
    
    data_f = space2batchLongForm(data_fake, offset)
    data_r = space2batchLongForm(data_real, offset)

    hue = np.concatenate([np.zeros((data_f.shape[0],)), np.ones((data_r.shape[0]))])
    data = np.concatenate((data_f, data_r), axis=0)

    dataWithHue = np.concatenate([data, hue[:,np.newaxis]],axis=1)
    logging.debug(dataWithHue.shape)
    
    grids = []

    for couple_idx, couple in enumerate(couples):
        if couple[0]!=couple[1]:
            var_couple = (variables[couple[0]], variables[couple[1]])
            data2plot = dataWithHue[:,(couple[0],couple[1],-1)]
            grids.append(sns.displot(pd.Dataframe(
                                {"data" : data2plot.ravel(),
                                "column" : [variables[couple[0]], variables[couple[1]], "origin"]
                                }), x=variables[couple[0]], y=variables[couples[1]], hue="origin", kind="kde")
                        )
    logging.debug(f"len of facet grids {len(grids)}")
    return grids
