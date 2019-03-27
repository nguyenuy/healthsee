#!/bin/bash

mkdir Data
curl https://data.cms.gov/api/views/f3yb-8552/rows.csv?accessType=DOWNLOAD --output Data/2018-12.csv

