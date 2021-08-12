# snippets

Lest we forget...

2021-08-09 - fast tropmask

```python
import xarray as xr
press=xr.open_dataset(disk+'xnktd_p.nc',  chunks = {'time': 12})
press['mask'] = xr.ones_like(press.p)
press['mask'] = press.mask.where(press.level_height<tropht.trop_hgt,other=0.)
press.mask.to_netcdf(jobid+'_mask.nc',encoding={"mask": {"dtype": "f8"}})
```

2021-08-09 - multiindex

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
