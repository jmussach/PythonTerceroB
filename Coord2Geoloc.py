__author__ = 'David'
from geopy.geocoders import Nominatim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim
geolocator = Nominatim()
location = geolocator.reverse("52.509669,13.376294")
print (location.address)