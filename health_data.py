# HealthScore Calculator Module

from datetime import datetime
from uszipcode import SearchEngine
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic


# Let's just load everything up front in this module
df_2018 = pd.read_csv('Data/2018-12.csv')
search = SearchEngine(simple_zipcode=False)

class Facilities:
    def __init__(self, df, zipcode, closest_zipcodes, given_loc):
        self.df = df
        self.zipcode = zipcode
        #self.facilities = df[df['ZIP_CD'].isin(closest_zipcodes)]
        self.active_fac_df = self.get_active_facilities()
        self.given_loc = given_loc
        self.closed_fac_df = self.get_closed_facilities()
        
    def get_active_facilities(self):
        ''' Retrieves active facilities
        '''
        facilities = self.df[self.df['ZIP_CD'].isin(closest_zipcodes)]
        return facilities[facilities['PGM_TRMNTN_CD']==0]
    
    def get_closed_facilities(self):
        facilities = self.df[self.df['ZIP_CD'].isin(closest_zipcodes)]
        return facilities[facilities['PGM_TRMNTN_CD']==2]
    
    def geolocate_lat_long(self, st_adr):
        fac_location = geolocator.geocode(st_adr)
        print(fac_location)
        if fac_location is None:
            return None, None, None
        fac_coor = (fac_location.latitude, fac_location.longitude)
        dist = geodesic(self.given_loc, fac_coor).miles
        print(dist)
        return fac_location.latitude, fac_location.longitude, dist
    
    def insert_lat_long_dist_columns(self):
        self.active_fac_df['COMPLETE_ADR'] = self.active_fac_df['ST_ADR'] + ', ' + self.active_fac_df['CITY_NAME'] + ', GA'
        self.active_fac_df['lat'], self.active_fac_df['long'], self.active_fac_df['Dist'] = zip(*self.active_fac_df['COMPLETE_ADR'].map(self.geolocate_lat_long))
        
    def insert_facility_age_column(self, colname):
        self.active_fac_df['ORGNL_PRTCPTN_DT'] = pd.to_datetime(self.active_fac_df['ORGNL_PRTCPTN_DT'], format='%Y%M%d')
        self.active_fac_df[colname] = datetime.today().year - self.active_fac_df['ORGNL_PRTCPTN_DT'].dt.year
    


''' General Module Functions
'''
def get_zipcode_info(zipcode):
    '''Returns a dict containing information for a zipcode
    '''
    return search.by_zipcode(zipcode).to_dict()

def get_zipcode_coordinates(zipcode):
    '''Gets the zipcodes coordinates as a tuple
    '''
    coordinates = search.by_zipcode(zipcode).to_dict()
    lat = coordinates['lat']
    lng = coordinates['lng']
    return (lat, lng)

def get_closest_zipcodes(zipcode):
    '''Gets the list of closest zipcodes within a 20 mile radius
    '''
    coordinates = get_zipcode_coordinates(zipcode)
    closest_zipcodes = search.by_coordinates(coordinates[0], coordinates[1], radius=5, returns=100)
    closest_zipcodes = list(map(lambda x: x.zipcode, closest_zipcodes))
    return closest_zipcodes


def get_zipcode_score(zipcode):
    general_zipcode_information = search.by_zipcode(zipcode)
    result = {}
    


# +
if __name__ == "__main__":
    zipcode = 30312
    slalom_loc = (33.854473,-84.360729)
    closest_zipcodes = get_closest_zipcodes(zipcode)

    zip_facilities = Facilities(df_2018, zipcode, closest_zipcodes, slalom_loc)

    print(zip_facilities.active_fac_df.shape)
    
    geolocator = Nominatim(user_agent="healthsea", timeout=5)
    zip_facilities.insert_lat_long_dist_columns()
    zip_facilities.insert_facility_age_column('age_of_facility')
    
    
    
    
# -
zip_facilities.active_fac_df = zip_facilities.active_fac_df.sort_values(['Dist','BED_CNT', 'RN_CNT', 'age_of_facility'], ascending=[True, True, True, True])
zip_facilities.active_fac_df[['FAC_NAME', 'Dist','BED_CNT', 'RN_CNT', 'age_of_facility']]
#EMPLEE_CNT









