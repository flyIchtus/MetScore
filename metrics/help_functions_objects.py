import os
import geopy.distance
import numpy as np
def test_and_create_path(path) :
     if not os.path.exists(path):
         os.system('mkdir -p ' + path)
     return "OK"

def zonage_bord(Zone) :
    Zone_l = ["NO","SO","SE","NE","C","France"]
    X_min = [230,250,540,460,360,250]
    Y_min = [300,150,120,300,220,120]
    index = Zone_l.index(Zone)
    return X_min[index],Y_min[index]

def grid_to_lat_lon(coord,X_min=0,Y_min=0):
    Lat_min = 37.5
    Lat_max = 55.4
    Lon_min = -12.0
    Lon_max = 16.0
    n_lat = 717
    n_lon = 1121
    Lat = Lat_min + (coord[1] + Y_min) * (Lat_max - Lat_min) / n_lat
    Lon = Lon_min + (coord[0] + X_min) * (Lon_max - Lon_min)/ n_lon
    return Lon,Lat

def distance_lat_lon(x_lat,x_lon,y_lat,y_lon):
    X = (x_lat, x_lon)
    Y = (y_lat, y_lon)
    Distance = geopy.distance.geodesic(X,Y).km
    return Distance

def merge_dicts(dict_list):
    """
    Given a list of dictionaries with shared attributes, 
    merge all those dictionaries by making arrays of attribute values for each attribute
    """
    merge_dic = {k : [] for k in dict_list[0].keys()}
    # trick : this uses for loops but with dictionaries this should be fast
    for d_idx, dic in enumerate(dict_list):
        for key, value in dic.items():
            # concatenating lists
            merge_dic[key] = merge_dic[key] + value
    for key in merge_dic.keys():
        merge_dic[key] = np.array(merge_dic[key])
    return merge_dic



