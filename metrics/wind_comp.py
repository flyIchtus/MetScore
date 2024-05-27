import math

import numpy as np
import pyproj

#########################
#    INITIALIZATION     #
#########################

_PROJ = {
    "GEO": "epsg:4326",
    "L2E": "epsg:27582 +es=0.0068034876 +rf=293.4660212940",  # 27572 L2 pas E
    "L93": "epsg:2154",
}


class ProjxError(Exception):
    """Toutes les erreurs de projx sont transmises sous cette exception."""

    pass


def giveProjection(projection):
    """
    Renvoie la projection sous la forme epsg:xxxx.
    @param projection : chaine de caractères indiquant un code epsg : "epsg:xxx"
    """
    if projection in _PROJ.keys():
        return _PROJ[projection]
    elif projection.startswith("epsg"):
        return projection
    else:
        raise (
            ProjxError,
            "Sytème de projection %s inconnu. %s supportés."
            % (projection, _PROJ.keys()),
        )


def giveConvAngle(xRef, yRef, projection):
    """
    Retourne l'angle de convergence des méridiens au point de coordonnées xRef yRef.
    @type xRef: réel
    @param xRef: Coordonnées en X.
    @type yRef: réel
    @param yRef: Coordonnées en Y.
    @type projection: string soit "L93" soit "L2E".
    @param projection:  Système des coordonnées.
    @rtype: réel
    @return: Angle en dégré de convergence des méridiens.
    """
    p = pyproj.Proj(init=giveProjection(projection))

    # Projection geo
    pGeo = pyproj.Proj(init=_PROJ["GEO"])

    # Projection des coordonnées de ref en geo
    xRefGeo, yRefGeo = pyproj.transform(p, pGeo, xRef, yRef)

    # Deplacement sur meridien nord.
    g = pyproj.Geod(ellps="WGS84")
    yGeoDelat1 = yRefGeo + 0.25
    (t1, t2, dist) = g.inv(xRefGeo, yRefGeo, xRefGeo, yGeoDelat1)
    xRefDelta1, yRefDelta1 = pyproj.transform(pGeo, p, xRefGeo, yGeoDelat1)
    yGeoDelat2 = yRefGeo - 0.25
    (t1, t2, dist2) = g.inv(xRefGeo, yRefGeo, xRefGeo, yGeoDelat2)
    xRefDelta2, yRefDelta2 = pyproj.transform(pGeo, p, xRefGeo, yGeoDelat2)

    # Calcul de l'angle
    tetha1 = math.degrees(math.asin((xRef - xRefDelta1) / dist))
    tetha2 = math.degrees(math.asin((xRefDelta2 - xRef) / dist2))

    return (tetha2 + tetha1) / 2.0


#########################
# FONCTIONS SUR LE VENT #
#########################


def angle_diff(beta1, beta2):
    a1 = beta1 - beta2
    a2 = 360 - (beta1 - beta2)
    a3 = -(beta1 - beta2 + 360)
    # logging.debug(np.abs(a1), np.abs(a2), np.abs(a3))
    a_min = np.minimum(np.abs(a1), np.abs(a2))
    a_min = np.minimum(a_min, np.abs(a3))
    # logging.debug(a_min.shape)
    # if a_min == abs(a1):
    #    alpha = a1
    # elif a_min == abs(a2):
    #    alpha = a2
    # elif a_min == abs(a3):
    #    alpha = a3
    a_min = np.where(np.abs(a1) > np.abs(a2), a2, a1)
    a_min = np.where(np.abs(a_min) > np.abs(a3), a3, a_min)
    return a_min


def computeWindDir(U, V, xRef=None, yRef=None, proj=None):
    """
    Calcul de la vitesse et la direction du vent à partir des composantes U et V
    @para U: vent zonal (m/s)
    @type U: float
    @param V: vent meridional (m/s)
    @type V: float
    @param xRef: longitude du point de reference (pour calculer la convergence des méridiens)
    @type xRef: float
    @param yRef: latitude du point de reference (pour calculer la convergence des méridiens)
    @type yRef: float
    @type projection: string soit "L93" soit "L2E", soit "WGS84" etc.
    @param projection:  Système des coordonnées.
    @rtype: tuple
    @return: vitesse (m/s) et direction du vent exprimée en degrés météo (0 = vent du nord).
    """
    ff = np.sqrt(U * U + V * V)

    dd3 = (180 + 180 / np.pi * np.arctan2(U, V)) % 360
    # logging.debug(dd3)

    # Prise en compte de la déclinaison des meridiens.
    if None not in (xRef, yRef, proj):
        devAng = giveConvAngle(xRef, yRef, proj)
        dd3 -= devAng

    return ff, dd3


def debiasing(X_p, real_ens_p, conditioning_members, mode=None):
    if mode == "SEED":
        N_a = int(X_p.shape[0] / conditioning_members)
        for i in range(int(conditioning_members)):
            Gan_avg_mem = np.mean(X_p[i * N_a : (i + 1) * N_a], axis=0)
            Bias = real_ens_p[i] - Gan_avg_mem
            Bias[1] = 0.0
            X_p[i * N_a : (i + 1) * N_a] = X_p[i * N_a : (i + 1) * N_a] + Bias
        return X_p
    elif mode == "ENSAVG":
        real_ens_p = real_ens_p[0:conditioning_members]  # ATTENTION
        Gan_avg = np.mean(X_p, axis=0)
        Real_ens_avg = np.mean(real_ens_p, axis=0)
        Bias = Real_ens_avg - Gan_avg
        Bias[1] = 0.0

        X_p = X_p + Bias
        return X_p

    else:
        raise Exception("Debiasing method not recognized")


def debiasing_multi_dates(X_p, real_ens_p):
    N_a = X_p.shape[1] // real_ens_p.shape[1]
    for i in range(real_ens_p.shape[1]):
        Gan_avg_mem = np.mean(X_p[:, i * N_a : (i + 1) * N_a], axis=1)
        Bias = real_ens_p[:, i] - Gan_avg_mem
        Bias[:, 1] = 0.0
        X_p[:, i * N_a : (i + 1) * N_a] = X_p[:, i * N_a : (i + 1) * N_a] + np.stack(
            [Bias for j in range(N_a)], axis=1
        )
    return X_p
