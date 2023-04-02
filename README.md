# snippets

Lest we forget...

## Jupyter
```python
!jupyter kernelspec list
```

```python 
!echo y | jupyter kernelspec uninstall geopandas-0.8-2022-10-20 
```

## Pandas
```python
#
# Drop NA from Dataframe
#
df.dropna(thresh=2)   #Drop row if it does not have at least two values that are **not** NaN
data2 = data.dropna(subset = ['review_scores_rating']) #drop row is column contains NaN

#
# remove data that are 3 s.d. away from mean
#

def zscore(sample):
    mean = sample.mean()
    std = sample.std()
    return (sample - mean) / std
    
to_drop = data[(zscore(data.square_feet)<3) & (zscore(data.price)<3)]

#
# scaling dataframe columns
#
# cont_vars = column names as list of strings
from sklearn.preprocessing import StandardScaler
stander = StandardScaler()
pd.DataFrame(stander.fit_transform(data[cont_vars]), columns=cont_vars)
```

Ignore warnings after first warning

```python
import warnings
warnings.filterwarnings(action='once') # https://stackoverflow.com/questions/9031783/hide-all-warnings-in-ipython 
```

### 2021-08-09 - Pandas multiindex

```python
latboxes = [[-90,-30],[-30,0],[0,30],[30,90]]
midpt_lat = np.zeros(len(latboxes))
for ivar in range(0,len(latboxes)):
  midpt_lat[ivar] = (latboxes[ivar][1] - latboxes[ivar][0])/2 + latboxes[ivar][0]
ann_index = pd.MultiIndex.from_product(iterables = [midpt_lat,  midpt_p, ann_vals.index],  names=['latbox_deg', 'pbox_hpa', 'time'])
ann_data = pd.DataFrame(np.zeros(ntimes//12*3*4), index=ann_index, columns=['oh'])
# assign a series to a set of indices
ann_data.loc[latmid, pmid/100., tmp_vals.index[ivar]]['oh'] = trends.data[ivar]
```

### 2022-07-17 - Pandas dataframe from Numpy vectors
```python

dataFrame = pd.DataFrame({'col1':np.arange(0,10), 'col2':np.arange(10,20)})
dataFrame = pd.DataFrame(zip(np.arange(0,10), np.arange(10,20)), columns=['col1','col2'])

```

# sane names

```python 
start_end_years = [[1950,1975], [1979,2014]]
for start_end_year in start_end_years:
    start_year = start_end_year[0]
    end_year = start_end_year[-1]
```

### MONSooN 

```bash
ssh USER@lander.monsoon-metoffice.co.uk "ssh xcsc \" cd && tar cf - NO_aircrft_anthropogenic_1849_2015_time_series.nc \" " | tar xvf -
```

## VSCODE - search for unique lines and delete duplicates
```
search ``` ^(.*)(\n\1)+$ ```

replace ``` $1 ```
```

# Xarray

```python

# create xarray data frame

xr.DataArray(ljw_aod.tas.data,
                            coords=[ljw_aod.time, ljw_aod.latitude, ljw_aod.longitude],
                            dims=['time', 'lat', 'lon'])
```

On chunking: [Dask pages](https://xarray.pydata.org/en/v0.10.2/dask.html)

With analysis pipelines involving both spatial subsetting and temporal resampling, dask performance can become very slow in certain cases. Here are some optimization tips we have found through experience:

1. Do your spatial and temporal indexing (e.g. .sel() or .isel()) early in the pipeline, especially before calling resample() or groupby(). Grouping and rasampling triggers some computation on all the blocks, which in theory should commute with indexing, but this optimization hasn’t been implemented in dask yet. (See dask issue #746).

2. Save intermediate results to disk as a netCDF files (using to_netcdf()) and then load them again with open_dataset() for further computations. For example, if subtracting temporal mean from a dataset, save the temporal mean to disk before subtracting. Again, in theory, dask should be able to do the computation in a streaming fashion, but in practice this is a fail case for the dask scheduler, because it tries to keep every chunk of an array that it computes in memory. (See dask issue #874)

3. Specify smaller chunks across space when using open_mfdataset() (e.g., chunks={'latitude': 10, 'longitude': 10}). This makes spatial subsetting easier, because there’s no risk you will load chunks of data referring to different chunks (probably not necessary if you follow suggestion 1).

## 2021-10-07 - two ways with significance testing

```python
def ks_mask_out_insig(cube1, cube2, nyears1, nyears2):
    from scipy.stats import ks_2samp
    import iris
    conf = 0.2
    lats = cube1.coord('latitude').points
    lons = cube2.coord('longitude').points
    store_statmask1=np.zeros([len(lats),len(lons)])
    store_base1 = cube1.data.data
    store_anom1 = cube2.data.data
    # If the difference between the baseline and anomaly ensembles is statistically significant
    # at the 95% confidence level according to a K-S test don't stipple, ie set mask = 0.
    for a in range(len(lats)):
        for b in range(len(lons)):
            (stat1,pval1)=ks_2samp(store_base1[:,a,b],store_anom1[:,a,b])
            if pval1<conf:
                store_statmask1[a,b]=1.0
    return store_statmask1
    
    
def masked_mean_3d(base_cube,expt_cube, t_cut):
    import scipy
    import scipy.stats
    import numpy as np

    P_test=0.05

    base=np.float64(base_cube.data[t_cut:,:,:])
    expt=np.float64(expt_cube.data[t_cut:,:,:])
    lon0=np.array(base_cube.coord('longitude').points,dtype=np.float64)
    lat=np.array(base_cube.coord('latitude').points,dtype=np.float64)

    T_vals,P_vals=scipy.stats.ttest_ind(base, expt, axis=0, equal_var=False)

    # If the p-value is greater than the threshold, then we accept the null hypothesis of equal averages and MASK OUT
    # see https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mstats.ttest_ind.html
    mask_out_for_insig=np.ones([len(lat), len(lon0)], dtype=np.float64)
    i=0
    while i < len(lat):
        j=0
        while j < len(lon0):
            if P_vals[i,j] > P_test:
#               print("!!!")
               mask_out_for_insig[i,j] = 0.
            j = j+1
        i = i+1
    return mask_out_for_insig
```

## 2021-08-09 - fast tropmask

```python
import xarray as xr
press=xr.open_dataset(disk+'xnktd_p.nc',  chunks = {'time': 12})
press['mask'] = xr.ones_like(press.p)
press['mask'] = press.mask.where(press.level_height<tropht.trop_hgt,other=0.)
press.mask.to_netcdf(jobid+'_mask.nc',encoding={"mask": {"dtype": "f8"}})
```

# plotting

2021-08-11 - categorical colorbar

```python
map = Basemap(projection='merc',  llcrnrlon=-120.,llcrnrlat=0.,urcrnrlon=70.,urcrnrlat=60.)
# convert 1D matrices into 2D fill matrices for processing:
x,y = (NA_toar_data.lon.data, NA_toar_data.lat.data)
xx, yy = np.meshgrid(x, y)
xx, yy = map(xx, yy)
map.drawcoastlines()
map.pcolormesh(xx, yy, TOAR_trend_sig, cmap=plt.cm.get_cmap('PuOr_r', 2), alpha=0.75)
target_names=['Not significant', 'Statistically significant']
# This function formatter will replace integers with target names
formatter = plt.FuncFormatter(lambda val, loc: target_names[val])
# We must be sure to specify the ticks matching our target names
plt.colorbar(ticks=[0, 1], format=formatter, shrink=0.5);
# Set the clim so that labels are centered on each block
plt.clim(-0.5, 1.5)
```


# older stuff

## Python

```python
import os

# The notifier function
def notify(title, subtitle, message):
    t = '-title {!r}'.format(title)
    s = '-subtitle {!r}'.format(subtitle)
    m = '-message {!r}'.format(message)
    os.system('terminal-notifier {}'.format(' '.join([m, t, s])))

# Calling the function
notify(title    = 'A Real Notification',
       subtitle = 'with python',
       message  = 'Hello, this is me, notifying you!')
```

## Shell

```bash
#!/bin/bash
for filename in $1*.nc; do
	ncatted -h -a contact,global,o,c,"Paul Griffiths paul.griffiths@ncas.ac.uk" ${filename}
	ncatted -h -a institution,global,o,c,"National Centre for Atmospheric Science" ${filename}
	ncatted -h -a project,global,o,c,"UKRI NERC C-CLEAR DTP, grant reference number:  NE/S007164/1" ${filename}
	ncatted -h -a model,global,o,c,"Data were generated using the UK Earth System Model version 1 (UKESM-1) at N96 horizontal resolution over global domain." ${filename}
	ncatted -h -a model_description,global,o,c,"Staniaszek et al. (2021) Climate and air quality benefits from a net-zeroanthropogenic methane emissions scenario" ${filename}
	ncatted -h -a run,global,o,c,"uby186 is a climate simulation for Year 2000 with methane emissions" ${filename}
	ncatted -h -a run_experiment,global,o,c,"NZAME - Year 2015-2050 with Net-Zero anthropogenic methane emissions" ${filename}
done
```

```bash
#!/bin/bash

# script to :
# generate maps of averaged monthly data
# generate maps of annual anomalies from all years' mean
# generate maps of standard deviation of monthly data from all years' mean


cd /work/n02/n02/ptg21/xin_data

model='xglgra'
stream='.pmk'
work_dir=/work/n02/n02/ptg21/xin_data
temp_dir=/work/n02/n02/ptg21/xin_data

cd ${work_dir}
echo '\nworking dir is \t'${work_dir}

year_array='1 2'
for zzYear in ${year_array}; do
	month_array='jan feb mar apr may jun jul aug sep oct nov dec'
	# generate a string containing files to be averaged, rms-ed 
	for zzMonth in ${month_array}; do
		files=${files}' '${model}${stream}${zzYear}${zzMonth}'.nc'
	done
done

all_file=${model}'_all.nc'
anom_file=${model}'_anom.nc'
ave_file=${model}'_ave.nc'
std_file=${model}'_std.nc'


echo '\nconcatenating the following files to form ' ${all_file}', anomalies and std dev'
echo 'generating averages over years' ${year_array} 'and months' ${month_array} '\n'

ncrcat -O ${files} ${all_file} 

# First construct the annual mean of tracer18 from the various input files
# need to specifiy explicitly time dimension?  apparently not

echo '\naveraging records \n'

ncra -O -v tracer18 ${all_file} ${ave_file}

for zzYear in ${year_array}; do
	echo 'Processing year '${zzYear}
	files=''
	annual_anom_file=${model}${stream}${zzYear}'_anom.nc'
	annual_mean_std_dev_file=${model}${stream}${zzYear}'_std_dev.nc'
	# generate a string containing files to be averaged, rms-ed 
	for zzMonth in ${month_array}; do
		file=${model}${stream}${zzYear}${zzMonth}'.nc'
		anom_file=${model}${stream}${zzYear}${zzMonth}'_anom.nc'
 		ncbo -O -v tracer18  ${file} ${ave_file} ${anom_file}
		files=${files}' '${anom_file}
	done
	# concatenate records to give a 12 month series of monthly anomalies from all-years' mean
	ncrcat -O ${files} ${annual_anom_file} 

	# generate the standard deviation from all years' mean
	# Note the use of `-y rmssdn' (rather than `-y rms')
	# the final step. This ensures the standard deviation 
	# is correctly normalized by one fewer than the number of time samples. 
	ncra -O -y rmssdn -v tracer18 ${annual_anom_file} ${annual_mean_std_dev_file}

	# average anomalies to give a year average anomaly from all years' mean
	ncra -O -v tracer18 ${annual_anom_file} ${annual_anom_file}
	
	# average standard deviations to give a year average standard deviation
	ncra -O -v tracer18 ${annual_mean_std_dev_file} ${annual_mean_std_dev_file}
done

echo '\ntidying up \n'
#tidy up by removing monthly dev files
for zzYear in ${year_array}; do
	 for zzMonth in ${month_array}; do
		file=${model}${stream}${zzYear}${zzMonth}'_anom.nc'
		rm ${file}
	done
done
```

remove dim_0 error in netCDF generated from malformed PP files - see [here](https://sourceforge.net/p/nco/discussion/9830/thread/3d35b42d/)

```sh
for zzFilename in u-bh443*_tas.nc; do
   [ -e "$zzFilename" ] || continue
   # ... rest of the loop body
  echo ${zzFilename} ${zzFilename}.fix
   ncks -O -x -v dim0,surface_temp_0,time_0,forecast_period_0_bnds,forecast_period_0,time_0_bnds ${zzFilename} ${zzFilename}.fix
done
```

fix time not being a record dimension in other Iris extracts

```sh
for zzFilename in u-br799*.nc; do
   [ -e "$zzFilename" ] || continue
   # ... rest of the loop body
		echo ${zzFilename} ${zzFilename}.fix
   ncks -O --mk_rec_dmn time ${zzFilename} ${zzFilename}.fix
done
```
