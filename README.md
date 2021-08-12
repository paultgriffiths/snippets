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
