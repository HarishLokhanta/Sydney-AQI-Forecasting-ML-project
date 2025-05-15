import numpy as np

_PM25_BP = [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
_NO2_BP  = [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
            (361, 649, 151, 200), (650, 1249, 201, 300),
            (1250, 1649, 301, 400), (1650, 2049, 401, 500)]

def _interp(c, bp):
    for lo, hi, ilo, ihi in bp:
        if lo <= c <= hi:
            return (ihi-ilo)/(hi-lo)*(c-lo)+ilo
    return np.nan

def aqi(pm, no2):
    """Return overall AQI (Australian variation)."""
    return max(_interp(pm, _PM25_BP),
               _interp(no2*0.522, _NO2_BP))