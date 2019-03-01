#!/bin/bash

mkdir Data
curl https://data.cms.gov/api/views/f3yb-8552/rows.csv?accessType=DOWNLOAD --output Data/2018-12.csv
curl https://data.cms.gov/api/views/u3qj-eifx/rows.csv?accessType=DOWNLOAD --output Data/2017-12.csv
curl https://data.cms.gov/api/views/7g3r-tu92/rows.csv?accessType=DOWNLOAD --output Data/2016-12.csv
curl https://data.cms.gov/api/views/xyt9-wu4y/rows.csv?accessType=DOWNLOAD --output Data/2015-12.csv
curl https://data.cms.gov/api/views/rh3v-57bs/rows.csv?accessType=DOWNLOAD --output Data/2014-12.csv
curl https://data.cms.gov/api/views/iem3-5cva/rows.csv?accessType=DOWNLOAD --output Data/2012-12.csv

