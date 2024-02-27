from astropy.convolution import convolve_fft,Gaussian2DKernel
import numpy as np
import os,pickle,datetime
import help_functions_objects as hlp
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes
import skimage.measure as skimage
from scipy import ndimage
from collections import namedtuple
#standard deviation (~ width in grid points) of the smoothing kernels for each type of rain objects
R_tot = 15
R_moderate = 12
R_heavy = 10
#smoothing kernels
Kernel_tt = Gaussian2DKernel(R_tot)
Kernel_moderate = Gaussian2DKernel(R_moderate)
Kernel_heavy = Gaussian2DKernel(R_heavy)


Zone = namedtuple("Zone", ["name", "X_min", "Y_min", "nb_lon", "nb_lat"])

def grid_to_lat_lon_C_mass(C_mass,zone):
    X_min, Y_min = zone.X_min, zone.Y_min
    Lat_min, Lon_min = hlp.grid_to_lat_lon(X_min,Y_min)
    Lat_max, Lon_max = hlp.grid_to_lat_lon(X_min + zone.nb_lon,Y_min + zone.nb_lat)
    Y,X = C_mass[0], C_mass[1]
    Lat = Lat_min + Y * (Lat_max - Lat_min) / zone.nb_lat
    Lon = Lon_min + X * (Lon_max - Lon_min) / zone.nb_lon
    return Lat, Lon

def Quantiles(field,reflc,num_objet) :
    indices_instance = np.where(field==num_objet)
    Reflc_instance = reflc[indices_instance]
    quants = np.quantiles(Reflc_instance,[0.25,0.9])
    return quants

def Contours(field,zone) :
    """
    Detect, for a given raw precipitation field, the object-types associated to "total rain" (rr>0.1), "moderate rain" (rr>2.8) and "heavy rain" (rr>6.5)
    Inputs :
        field : np.array, raw precipitation field of shape H x W
        Zone :  an instance of Zone type (namedtuple)
    Returns:
        dict containing binarized versions of the raw field for each object type
    """
    nb_lon = zone.nb_lon
    nb_lat = zone.nb_lat
    global Kernel_tt
    global Kernel_moderate
    global Kernel_heavy
    Object_RR_tt = np.zeros((nb_lat,nb_lon))
    Object_RR_moderate = np.zeros((nb_lat,nb_lon))
    Object_RR_heavy = np.zeros((nb_lat,nb_lon))
    if np.max(field)>=0.1 : #pas besoin de faire convolve_fft si pluies pas assez fortes
        Conv_R_tot = convolve_fft(field,Kernel_tt)
        Object_RR_tt[Conv_R_tot>0.2] = 1.
        Object_RR_tt[Conv_R_tot<=0.2] = 0.

    if np.max(field)>=2.8 :
        Conv_R_moderate = convolve_fft(field,Kernel_moderate)
        Object_RR_moderate[Conv_R_moderate>3.] = 1.
        Object_RR_moderate[Conv_R_moderate<=3.] = 0.
    if np.max(field)>=6.5 :
        Conv_R_heavy = convolve_fft(field,Kernel_heavy)
        Object_RR_heavy[Conv_R_heavy>7.] = 1.
        Object_RR_heavy[Conv_R_heavy<=7.] = 0.

    return {"total" : Object_RR_tt, "moderate" : Object_RR_moderate, "heavy": Object_RR_heavy}

def attributesCore(field,reflc,zone):
    """
    Attribut extraction for precipitation field given specific objects
    Inputs:
        field : np.array of integers, shape H x W, containing binary representations 
                of all instances from a given object type (total XOR moderate XOR heavy)
        reflc : np.array of floats, shape H x W, containing continous precipitation ("raw" field)
        Zone :  an instance of Zone type (namedtuple)
    Outputs:
        attributes : dict containing, for each attribute, list of value for this attribute of all the detected object instances in the field
    """
    attributes = {}
    attributes["num_objet"] = [] #Object id
    attributes["Area"] = []
    attributes["Q90"] = []
    attributes["Q25"] = []
    attributes["Y_Lat"] = []
    attributes["X_Lon"] = []

    #field contain no object of the type object_type --> empty dictionary
    if np.max(field) < 1 :
        return attributes
    # field contain at least 1 object
    else :
        # counting instances of the object_type and returning a labelled field from binary field
        field_m,num = skimage.label(field, return_num=True)
        unique, counts = np.unique(field_m, return_counts=True)
        Liste_object_instances = dict(zip(unique, counts))

        #computing objects center of mass
        C_mass = ndimage.measurements.center_of_mass(field,field_m,index=list(range(1,num+1)))
        for instance in list(Liste_object_instances.keys())[1:] : #First instance = area without precipitation
            attributes["num_objet"].append(int(instance))
            attributes["Area"].append(int(Liste_object_instances[instance]))

            quants = Quantiles(field_m,reflc,instance)
            attributes["Q25"].append(quants[0])
            attributes["Q90"].append(quants[1])

            Lat,Lon = grid_to_lat_lon_C_mass(C_mass[instance-1],zone)
            attributes["X_Lon"].append(Lon)
            attributes["Y_Lat"].append(Lat)
        return attributes

def batchAttributes(batch, zone):
    objects_attributes = {"total": [], "moderate" : [], "heavy" : []}
    for sample_idx in range(batch.shape[0]):
         data = batch[i][0]
         objects = Contours(data,zone)
         for object_type in objects.keys():
            attributes = attributesCore(objects[object_type], data, zone)
            objects_attributes[object_type].append(attributes)   
    object_attributes['total'] = hlp.merge_dicts(object_attributes['total'])
    object_attributes['moderate'] = hlp.merge_dicts(object_attributes['moderate'])
    object_attributes['heavy'] = hlp.merge_dicts(object_attributes['heavy'])

    return objects_attributes



