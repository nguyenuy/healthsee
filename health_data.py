# HealthScore Calculator Module

from datetime import datetime
from uszipcode import SearchEngine
import numpy as np
import pandas as pd


# Let's just load everything up front in this module
df_2018 = pd.read_csv('Data/2018-12.csv')
search = SearchEngine(simple_zipcode=False)

class Facilities:
    def __init__(self, df, zipcode, closest_zipcodes):
        self.zipcode = zipcode
        self.facilities = df[df['ZIP_CD'].isin(closest_zipcodes)]

    def get_active_facilities(self):
        ''' Retrieves active facilities
        '''
        return self.facilities[self.facilities['PGM_TRMNTN_CD']==0]
    


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
    closest_zipcodes = search.by_coordinates(coordinates[0], coordinates[1], radius=20, returns=1000)
    closest_zipcodes = list(map(lambda x: x.zipcode, closest_zipcodes))
    return closest_zipcodes


def get_zipcode_score(zipcode):
    general_zipcode_information = search.by_zipcode(zipcode)
    result = {}


if __name__ == "__main__":
    zipcode = 30312
    closest_zipcodes = get_closest_zipcodes(zipcode)

    zip_facilities = Facilities(df_2018, zipcode, closest_zipcodes)

    df = zip_facilities.get_active_facilities()

    print(df.count())
    
