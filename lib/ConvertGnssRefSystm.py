from pyproj import Transformer
import pymap3d as pm

def dms2decimal(dms):
    '''
    dms = ((deg_num, deg_den), (min_num, min_den), (sec_num, sec_den))
    DD = decimal result (degrees)
    '''
    deg_num = dms[0][0]
    deg_den = dms[0][1]
    min_num = dms[1][0]
    min_den = dms[1][1]
    sec_num = dms[2][0]
    sec_den = dms[2][1]

    DD = deg_num/deg_den + min_num/min_den/60 + sec_num/sec_den/60/60

    return DD


def CovertGnssRefSystm(lat_dms, lon_dms, alt_frac, input_system  = 'epsg:6704', output_system = 'epsg:6706', enu_reference_point = (4347718.9572, 856411.9883, 4573643.2406)):
    '''
    lat_dms, lon_dms, alt_dms in degrees minutes seconds in fractional format
    '''
    lat = dms2decimal(lat_dms)
    lon = dms2decimal(lon_dms)
    alt = alt_frac[0]/alt_frac[1]

    transformer = Transformer.from_crs(input_system, output_system)
    transform_object = transformer.itransform([(enu_reference_point[0], enu_reference_point[1], enu_reference_point[2])])
    for pt in transform_object:
        enu_reference = (pt[0], pt[1], pt[2])

    enuX, enuY, enuZ = pm.geodetic2enu(lat, lon, alt, enu_reference[0], enu_reference[1], enu_reference[2])

    return enuX, enuY, enuZ