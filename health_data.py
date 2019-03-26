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
search_simple = SearchEngine(simple_zipcode=True)

# This is a long initial load...load all information into a dict and never use the zipcode database again for information
all_zipcodes_info = search.by_population(lower=0, upper=9999999999999, returns=40000)
all_zipcodes_info = list(map(lambda x: x.to_dict(), all_zipcodes_info))
all_zipcodes = list(map(lambda x: x['zipcode'], all_zipcodes_info))
all_zipcodes_info = {d['zipcode']: d for d in all_zipcodes_info}

class Facilities:
    def __init__(self, df, zipcode, closest_zipcodes):
        self.df = df
        self.zipcode = zipcode
        self.all_facilities = df[df['ZIP_CD'].isin(closest_zipcodes)]
        self.active_fac_df = self.get_active_facilities(closest_zipcodes)
        self.closed_fac_df = self.get_closed_facilities(closest_zipcodes)
        self.given_loc = get_zipcode_coordinates(zipcode)
        
    def get_active_facilities(self, closest_zipcodes):
        ''' Retrieves active facilities
        '''
        facilities = self.df[self.df['ZIP_CD'].isin(closest_zipcodes)]
        return facilities[facilities['PGM_TRMNTN_CD']==0]
    
    def get_closed_facilities(self, closest_zipcodes):
        '''Retrieves closed facilities, when there exists an termination date on a row
        '''
        facilities = self.df[self.df['ZIP_CD'].isin(closest_zipcodes)]
        return facilities[facilities['TRMNTN_EXPRTN_DT'].notnull()]
    
    def geolocate_lat_long(self, st_adr):
        fac_location = geolocator.geocode(st_adr)
        #print(fac_location)
        if fac_location is None:
            return None, None, None
        fac_coor = (fac_location.latitude, fac_location.longitude)
        dist = geodesic(self.given_loc, fac_coor).miles
        #print(dist)
        return fac_location.latitude, fac_location.longitude, dist
    
    def insert_lat_long_dist_columns(self):
        '''Inserts the Latitude and Longitude corresponding to an ACTIVE medical facility
           TODO: Speed up the API calls here. This is a bottleneck
        '''
        self.active_fac_df['COMPLETE_ADR'] = self.active_fac_df['ST_ADR'] + ', ' + self.active_fac_df['CITY_NAME'] + ', GA'
        self.active_fac_df['lat'], self.active_fac_df['long'], self.active_fac_df['Dist'] = zip(*self.active_fac_df['COMPLETE_ADR'].map(self.geolocate_lat_long))
        
    def insert_facility_age_column(self, colname='age_of_facility'):
        '''Calculates the Facility age and creates a new column with this calculation
        '''
        active_facilities = self.active_fac_df
        active_facilities['ORGNL_PRTCPTN_DT'] = pd.to_datetime(active_facilities['ORGNL_PRTCPTN_DT'], format='%Y%M%d')
        self.active_fac_df[colname] = datetime.today().year - active_facilities['ORGNL_PRTCPTN_DT'].dt.year

    def get_number_of_beds(self):
        '''Gets the total number of beds from the list of active facilities
           The focus is here on BED_CNT. The dataset has multiple BED_CNT fields but we don't differentiate them here
        '''
        return self.active_fac_df['BED_CNT'].sum() 

    def get_number_of_closures_past_ten_years(self):
        '''Year closure is determined by TRMNTN_EXPRTN_DT and is represented by floating value like '19981231.0'
           This function obtains the net closures of facilities
           negative closings => more openings yay
        '''
        ten_year_ago = datetime.today().year - 10
        ten_year_ago = ten_year_ago * 10000.0

        closures = self.closed_fac_df
        closures = closures[closures['TRMNTN_EXPRTN_DT'] >= ten_year_ago]
        
        openings = self.active_fac_df
        openings = openings[openings['ORGNL_PRTCPTN_DT'] >= ten_year_ago]

        net_closures = len(closures.index) - len(openings.index)
        return net_closures
        
    


''' General Module Functions
'''
def get_zipcode_info(zipcode):
    '''Returns a dict containing information for a zipcode
    '''
    # Caching here to some degree
    try:
        return all_zipcodes_info[str(zipcode)]
    except:
        info = search.by_zipcode(zipcode).to_dict()
        return info

def get_zipcode_coordinates(zipcode):
    '''Gets the zipcodes coordinates as a tuple
    '''
    #coordinates = search.by_zipcode(zipcode).to_dict()
    coordinates = get_zipcode_info(zipcode)
    lat = coordinates['lat']
    lng = coordinates['lng']
    return (lat, lng)

def get_closest_zipcodes(zipcode, radius=5):
    '''Gets the list of closest zipcodes within a default=5 mile radius
    '''
    coordinates = get_zipcode_coordinates(zipcode)
    # import time
    #start_time = time.time()
    closest_zipcodes = search_simple.by_coordinates(coordinates[0], coordinates[1], radius, returns=100)
    closest_zipcodes = list(map(lambda x: x.zipcode, closest_zipcodes))
    #print("--- %s seconds ---" % (time.time() - start_time))
    return closest_zipcodes


def get_zipcode_score_and_data(zipcode):
    general_zipcode_information = search.by_zipcode(zipcode)

    healthscore = calculate_health_score(zipcode)


    result = {}

def set_metric_information(metric, description, value):
    pass

def calculate_health_score(zipcode):
    metrics = {}
    closest_zipcodes = get_closest_zipcodes(zipcode, radius=10)
    zip_facilities = Facilities(df_2018, zipcode, closest_zipcodes)

    # Metric 1: Closures past 10 years
    closures = zip_facilities.get_number_of_closures_past_ten_years()
    closures_metric = closures * -1
    metrics['closures'] = closures_metric

    # Metric 2: Population/Bed Count
    num_beds = zip_facilities.get_number_of_beds()
    total_population = 0
    for close_zip in closest_zipcodes:
        zipcode_info = get_zipcode_info(close_zip)
        total_population = total_population + zipcode_info['population']

    if num_beds >= 0.0:
        metrics['people_per_bed'] = total_population/num_beds
    else:
        metrics['people_per_bed'] = 0


    metrics['zipcode'] = zipcode
    return metrics

def build_metric_df(zipcodes):
    health_metrics = list(map(lambda x: calculate_health_score(x), zipcodes))

    metric_df = pd.DataFrame(health_metrics)
    metric_df = metric_df.set_index('zipcode')

    # Normalize metric columns
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler() 
    column_names_to_normalize = ['closures', 'people_per_bed']
    x = metric_df[column_names_to_normalize].values
    x_scaled = min_max_scaler.fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = metric_df.index)
    metric_df[column_names_to_normalize] = df_temp


    #print(metric_df.to_string())
    return metric_df


    

if __name__ == "__main__":
    # TODO: cache closest zip codes column that is precomputed in csv

    ###
    zipcode = 60629
    slalom_loc = (33.854473,-84.360729)
    closest_zipcodes = get_closest_zipcodes(zipcode)

    zip_facilities = Facilities(df_2018, zipcode, closest_zipcodes)
    print("NET CLOSURES PAST 10 YEARS: " + str(zip_facilities.get_number_of_closures_past_ten_years()))
    print("BED COUNT: " + str(zip_facilities.get_number_of_beds()))
    import json 
    print("METRICS: " + json.dumps(calculate_health_score(zipcode)))
    

    build_metric_df(all_zipcodes)
    # NOTE: This takes a while to run because it makes remote API calls to obtain the address
    exit(0)
    geolocator = Nominatim(user_agent="healthsea", timeout=5)
    zip_facilities.insert_lat_long_dist_columns()
    zip_facilities.insert_facility_age_column('age_of_facility')
    
    
    
    
# -
#zip_facilities.active_fac_df = zip_facilities.active_fac_df.sort_values(['Dist','BED_CNT', 'RN_CNT', 'age_of_facility'], ascending=[True, True, True, True])
#zip_facilities.active_fac_df[['FAC_NAME', 'Dist','BED_CNT', 'RN_CNT', 'age_of_facility']]
#EMPLEE_CNT









