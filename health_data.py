# HealthScore Calculator Module

from datetime import datetime
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from uszipcode import SearchEngine
import ast
import numpy as np
import pandas as pd


# Let's just load everything up front in this module
print('Loading in data.....')
df_2018 = pd.read_csv('Data/2018-12.csv', dtype={'zipcode': str})
df_closest_zipcodes = pd.read_csv('closest_zips.csv')
zip_code_list = df_closest_zipcodes['zipcode'].values.tolist()
df_closest_zipcodes = df_closest_zipcodes.set_index('zipcode').T
#search = SearchEngine(simple_zipcode=False)
#search_simple = SearchEngine(simple_zipcode=True)

print('Loading in pre-computed metrics')
df_metrics_normalized = pd.read_csv('normalized_metrics.csv', dtype={'zipcode': str})
df_metrics_normalized = df_metrics_normalized.set_index('zipcode').T

# This is a long initial load...load all information into a dict and never use the zipcode database again for information
print('Loading in all zipcodes')
#all_zipcodes_info = search.by_population(lower=0, upper=9999999999999, returns=40000) # This line runs extremely slowly
all_zipcodes = None
all_zipcodes_info = None

print('Done loading zipcode information')
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

    def get_average_age_facility(self):
        self.insert_facility_age_column()
        return self.active_fac_df['age_of_facility'].mean()

    def get_number_of_physicians(self):
        return self.active_fac_df['PHYSN_CNT'].sum()

    def get_number_of_facilities_offering_cancer_detection(self):
        nuclear_medicine_offerings = self.active_fac_df[self.active_fac_df['NUCLR_MDCN_SRVC_CD']!=0]
        diagnostic_radiology = self.active_fac_df[self.active_fac_df['DGNSTC_RDLGY_SRVC_CD']!=0]
        ct_scan = self.active_fac_df[self.active_fac_df['CT_SCAN_SRVC_CD']!=0]

        total = pd.concat([nuclear_medicine_offerings, diagnostic_radiology, ct_scan]).drop_duplicates()

        return len(total.index)

    def get_number_of_registered_nurses(self):
        return self.active_fac_df['RN_CNT'].sum()

    def get_number_of_registered_pharmacists(self):
        return self.active_fac_df['REG_PHRMCST_CNT'].sum()

    def get_number_of_unique_provider_categories(self):
        """
        Values of PRVDR_CTGRY_CD -- total of 18:
            01=Hospital
            02=Skilled Nursing Facility/Nursing Facility (Dually Certified)
            03=Skilled Nursing Facility/Nursing Facility (Distinct Part)
            04=Skilled Nursing Facility
            05=Home Health Agency
            06=Psychiatric Residential Treatment Facility
            07=Portable X-Ray Supplier
            08=Outpatient Physical Therapy/Speech Pathology
            09=End Stage Renal Disease Facility
            10=Nursing Facility
            11=Intermediate Care Facility/Individuals with Intellectual Disabilities
            12=Rural Health Clinic
            14=Comprehensive Outpatient Rehab Facility
            15=Ambulatory Surgical Center
            16=Hospice
            17=Organ Procurement Organization
            19=Community Mental Health Center
            21=Federally Qualified Health Center
        :return:
        """

        return len(df_2018["PRVDR_CTGRY_CD"].value_counts().index)

    def urgent_care_provider(self):
        """
        get ER, Urgent Care providers
        """
        er_hospitals = self.active_fac_df[(self.active_fac_df['PRVDR_CTGRY_CD'] == 1) &
                                          (self.active_fac_df['DCTD_ER_SRVC_CD'] >= 1.0)] \
            .drop_duplicates(subset=['FAC_NAME'])
        return er_hospitals['FAC_NAME'].values.tolist()

    def urgent_care_pediatric_provider(self):
        """
        get ER, Urgent Care providers offered for pediatrics
        """

        er_pediatric_hospitals = self.active_fac_df[(self.active_fac_df['PRVDR_CTGRY_CD'] == 1) &
                                          (self.active_fac_df['PED_SRVC_CD'] >= 1.0)] \
            .drop_duplicates(subset=['FAC_NAME'])
        return er_pediatric_hospitals['FAC_NAME'].values.tolist()


count = 0

''' General Module Functions
'''
def get_zipcode_info(zipcode):
    '''Returns a dict containing information for a zipcode
    '''
    # Caching here to some degree
    search = SearchEngine(simple_zipcode=False)
    search_simple = SearchEngine(simple_zipcode=True)
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
    search = SearchEngine(simple_zipcode=False)
    search_simple = SearchEngine(simple_zipcode=True)
    coordinates = get_zipcode_coordinates(zipcode)
    closest_zipcodes = search_simple.by_coordinates(coordinates[0], coordinates[1], radius, returns=100)
    closest_zipcodes = list(map(lambda x: x.zipcode, closest_zipcodes))
    return closest_zipcodes

def calculate_health_score(zipcode):
    metrics = {}
    closest_zipcodes = get_closest_zipcodes(zipcode, radius=20)
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

    if num_beds > 0.0:
        metrics['people_per_bed'] = total_population/num_beds
    else:
        metrics['people_per_bed'] = 0.0

    # Metric 3: Age of facility
    #zip_facilities.insert_facility_age_column()
    #metrics['avg_facility_age'] = np.mean(zip_facilities.active_fac_df['age_of_facility'])
    
    metrics['zipcode'] = zipcode

    # Metric 3: Average Age of Facility
    #average_age_facility = zip_facilities.get_average_age_facility()
    #if average_age_facility > 0.0:
    #    metrics['average_age_facility'] = 1.0/average_age_facility # Want the inverse to give newer facilities higher ranking
    #else:
    #    metrics['average_age_facility'] = 0.0
    
    # Metric 4: Population / On-Hand_Staff (Doctors)
    number_physicians = zip_facilities.get_number_of_physicians()
    if number_physicians > 0.0:
        metrics['people_per_physician'] = total_population/number_physicians
    else:
        metrics['people_per_physician'] = 0.0
    

    # Metric 5: Number of Facilities offering Cancer Detection Services
    metrics['facilities_offering_cancer_detect'] = zip_facilities.get_number_of_facilities_offering_cancer_detection()

    # Metric 6: Number of unique provider types
    metrics['num_of_types_of_facilities'] = zip_facilities.get_number_of_unique_provider_categories()

    # Metric 7: People per registered nurse
    number_nurses = zip_facilities.get_number_of_registered_nurses()
    if number_nurses > 0.0:
        metrics['people_per_registered_nurse'] = total_population/number_nurses
    else:
        metrics['people_per_registered_nurse'] = 0.0

    # Metric 8: People per registered pharmacist
    number_pharmacists = zip_facilities.get_number_of_registered_nurses()
    if number_nurses > 0.0:
        metrics['people_per_registered_pharmacist'] = total_population/number_pharmacists
    else:
        metrics['people_per_registered_pharmacist'] = 0.0

    # Metric 9: ER, Urgent Care hospitals
    urgent_care_hospitals = zip_facilities.urgent_care_provider()
    metrics['number_of_urgent_care'] = len(urgent_care_hospitals)
    #metrics['urgent_care_hospitals'] = urgent_care_hospitals

    # Metric 10: ER, Urgent Care hospitals - Pediatrics

    urgent_care_pediatrics_hospitals = zip_facilities.urgent_care_pediatric_provider()

    metrics['number_of_urgent_care_pediatrics'] = len(urgent_care_pediatrics_hospitals)
    #metrics['urgent_care_hospitals_pediatrics'] = urgent_care_pediatrics_hospitals

    global count
    count = count + 1
    if count % 500 == 0:
        print(str(datetime.now()) + ' ==> ' + str(count))
        
    return metrics




def build_metric_df(zipcodes):
    #health_metrics = list(map(lambda x: calculate_health_score(x), zipcodes))
    health_metrics = [calculate_health_score(x) for x in zipcodes]

    metric_df = pd.DataFrame(health_metrics)
    metric_df = metric_df.set_index('zipcode')

    # Normalize metric columns
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler()
    column_names_to_normalize = ['closures',
                                 'people_per_bed',
                                 'people_per_physician',
                                 'facilities_offering_cancer_detect',
                                 'num_of_types_of_facilities',
                                 'people_per_registered_nurse',
                                 'people_per_registered_pharmacist',
                                 'number_of_urgent_care',
                                 'number_of_urgent_care_pediatrics']
    x = metric_df[column_names_to_normalize].values
    x_scaled = min_max_scaler.fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = metric_df.index)
    metric_df[column_names_to_normalize] = df_temp


    #print(metric_df.to_string())
    metric_df.to_csv('normalized_metrics.csv')
    return metric_df

def get_total_normalized_health_score(zipcode):
    '''Gets the normalized health score for a zipcode
    '''
    'Puts this on a scale from 0 to 100'
    return df_metrics_normalized[str(zipcode)].sum()

def get_normalized_metrics(zipcode):
    '''Gets the normalized metrics for a zipcode as a dict
    '''
    return df_metrics_normalized[str(zipcode)].to_dict()

def get_similar_zipcodes(zipcode):
    '''Gets similar zipcodes with similar
    '''
    zipcode_score = get_total_normalized_health_score (zipcode)
    summed = df_metrics_normalized.sum()

    lower_bound = .95 * zipcode_score
    upper_bound = 1.05 * zipcode_score

    return summed.where((summed > lower_bound) & (summed < upper_bound )).dropna().keys().to_list()

def get_lowest_scores_zipcodes(number=10):
    summed = df_metrics_normalized.sum()
    return summed.nsmallest(number).keys().to_list()
    
def get_highest_scores_zipcodes(number=10):
    summed = df_metrics_normalized.sum()
    return summed.nlargest(number).keys().to_list()

def load():
    print('Now we are really loading things.........')
    global all_zipcodes_info
    global all_zipcodes
    search = SearchEngine(simple_zipcode=False)
    search_simple = SearchEngine(simple_zipcode=True)
    all_zipcodes_info = list(map(lambda x: search.by_zipcode(x).to_dict(), zip_code_list))
    all_zipcodes = list(map(lambda x: x['zipcode'], all_zipcodes_info))
    all_zipcodes_info = {d['zipcode']: d for d in all_zipcodes_info}


def get_zipcode_dict(zipcode):
    zipcodes = get_closest_zipcodes(zipcode, 20)
    blah = {}
    blah['zipcodes'] = zipcode
    blah['closest_zips'] = zipcodes
    return blah
def write_closest_zipcodes_csv():
    

    blah = [get_zipcode_dict(x) for x in all_zipcodes]
    df = pd.DataFrame(blah)
    df.to_csv('closest_zips.csv')



if __name__ == "__main__":
    load()
    print(get_similar_zipcodes('30084'))
    print(get_highest_scores_zipcodes())
    print(get_lowest_scores_zipcodes())
    print(get_normalized_metrics('33172'))
    print(get_total_normalized_health_score('33172'))
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
# zip_facilities.active_fac_df = zip_facilities.active_fac_df.sort_values(['Dist','BED_CNT', 'RN_CNT', 'age_of_facility'], ascending=[True, True, True, True])
# zip_facilities.active_fac_df[['FAC_NAME', 'Dist','BED_CNT', 'RN_CNT', 'age_of_facility']]
# EMPLEE_CNT









