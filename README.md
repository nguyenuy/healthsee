# healthSEE

## Getting Started
1. Set up a virtualenv with and activate with the following commands
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2. Install dependencies by running `pip install -r requirements.txt` in the root of the repo
3. Bring up a notebook session by running `jupyter notebook`
4. Execute the `download_data.sh` script to download the csv data into the Data directory. The data in this script is the **Provider of Services** dataset with more detail below

## Datasets Information
### Provider of Services
[Link to dataset](https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/Provider-of-Services/index.html)

### Zipcodes
This is a [library](https://pypi.org/project/uszipcode/) import that we are utilizing. It helps us identify nearby zipcodes given a latitude and longitude location and also extraneous information about the zipcode itself.