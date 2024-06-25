import skimage.measure as skimage
from scipy import ndimage

field_m,num = skimage.label(field, return_num=True) #identification unique des objets
C_mass = ndimage.measurements.center_of_mass(field,field_m,index=list(range(1,num + 1)))

