from glob import glob
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.signal import medfilt
import pandas as pd
import os
import sa_common
import argparse
from dask.distributed import LocalCluster, wait
import sys
import traceback


u4max = np.iinfo(np.uint32).max


def compress_all(nc_grids, min_dims=1):
    for var in nc_grids:
        if len(nc_grids[var].dims) >= min_dims:
            nc_grids[var].encoding["zlib"] = True
            nc_grids[var].encoding["complevel"] = 5
            nc_grids[var].encoding["contiguous"] = False
    return nc_grids


def correct_micros(micros, fs, tol=0.1, window=7):
    """
    micros is the timing data (integer microseconds) to be corrected.
    fs is the nominal sample rate in samples/second
    tol is the fraction of the sample interval used in defining large errors
    window is the number of samples used to correct single-point noise.
    
    Output is corrected so that the time change between samples is
    not greater than tol/fs.
    
    Two things can result in time that does not increase steadily.
    1. Random read errors, which may be positive or negative. 
        These are embedded in a steadily increasing ramp, so the 
        median of these samples (for a large enough window)
        Within window we expect these erros to be significantly differnt from the median
    2. Integer rollover, which results in an instananeous change of -u4_int_max

    First, run a median filter to eliminate most spikes. If the number of errors in the
    window is larger than (window-1)/2, then the window will have an incorrect value.
    But this value will be much smaller than the integer max, allowing us to isolate the
    truly large negative changes from integer rollovers.
    
    >>> test_ramp = np.arange(-25, 25).astype('u4') + u8max
    >>> medfilt(test_ramp, 3)
    Near the jump this code gives 
    [4294967292, 4294967293, 4294967294, 4294967294, 1, 1, 2, 3]
    i.e., the delta remains large.

    At the edges of the dataset, we have to be more careful. 
    After correcting the middle of the dataset, check for bad data one more time.
    If found, simply extrapolate using the floor of the mean sample rate.
    
    """
    nominal_dt_micro = int(1e6/fs)
    
    def get_bad(m):
        change = np.diff(m)
        return (change < 0) & (change > int(tol*nominal_dt_micro))

    # Median filter, and promote to 64 bit int
    med_micro = medfilt(micros, window).astype('i8')
    
    ### Detect the large negatives, and add those to the original data
    
    # After median filtering, the jump could be corrupted by noise.
    # Allow some tolerance
    big_neg_thresh = -int((1-tol)*u4max)
    sample_delta = np.diff(med_micro)
    # print(big_neg_thresh, "negative rollover threshold; ", sample_delta.min(), "lowest delta")
    large_neg = np.hstack([[False], 
                           (sample_delta < big_neg_thresh)])
    n_rollovers = large_neg.sum()
    # print(n_rollovers, "total rollovers")
    
    # Set the points with large negative deltas to uint4max,
    # then cumsum will accumulate correction amount to all points
    # after each detected jump.
    correction_mark = np.zeros_like(micros)
    correction_mark[large_neg] = u4max
    neg_correct = np.cumsum(correction_mark)
    
    micros = med_micro + neg_correct
    
    ### Check for remaining problems

    bad = get_bad(micros)

    if bad.sum() > 0:
        # medfilt one more time
        micros = medfilt(micros, window)
    else:
        return micros, n_rollovers

    bad = get_bad(micros)
    
    # Check for problems at the beginning and end
    front = slice(0, window)
    if bad[front].sum() > 0:
        micros[front] = micros[window] + nominal_dt_micro*np.arange(-window, 0, 1)

    back = slice(-window, None)
    if bad[back].sum() > 0:
        micros[back] = micros[-window] + nominal_dt_micro*np.arange(0, window, 1)
    
    # Now look at any remaning deltas
    bad = get_bad(micros)

    if bad.sum() > 0:
        # medfilt one more time with a very large window
        micros = medfilt(micros, window*3)
        bad = get_bad(micros)
        if bad.sum() > 0:
            raise ValueError("Could not correct all time errors; total bad = {0}".format(bad.sum()))
        else:
            return micros, n_rollovers
    else:
        return micros, n_rollovers


def interpolate_across_system_times(ds, SAMPLE_RATE=9600):
    
    nominal_dt = 1.0/SAMPLE_RATE
    good_sample_thresh = (3*nominal_dt)

    bad_adc = (ds['ADC'].values < 0.0) 
    adc_minus_system = (ds['dt_adc']-ds['dt_system']).values
    adc_minus_system[bad_adc] = np.nan
    jump = np.diff(adc_minus_system, prepend=np.nan)

    # Also blank out the ADC and ADC microseconds where we have bad data.
    ds['dt_adc'] = ds.dt_adc.where(~bad_adc, other=np.nan)
    ds['dt_system'] = ds.dt_system.where(~bad_adc, other=np.nan)
    
    is_jump = np.abs(jump) > good_sample_thresh
    jump_idx, = np.where(is_jump)
    # Include first and last points
    jump_idx = np.insert(jump_idx, [0,jump_idx.shape[0]], [0,adc_minus_system.shape[0]-1])
    jump = jump[jump_idx]
    
    all_samples = np.arange(adc_minus_system.shape[0])
    offset_curve = np.interp(all_samples,
                             jump_idx,
                             adc_minus_system[jump_idx])
    correction = offset_curve - adc_minus_system
    ds['time'] = ds.time_uncorrected - (correction*1e9).astype('timedelta64[ns]')
    return ds


def add_other_time_vars(ds):
    ds['dt_system'] = (ds.time_uncorrected-ds.time_uncorrected[0]).astype('timedelta64[ns]').astype('f8')/1e9
    ds['dt_adc'] = (ds.pps_micro - ds.pps_micro[0]).astype('f8')/1e6
    return ds


def read_single_file(file_info, sample_rate, total_rollovers=0, previous_filepath=None):
    data_packets = sa_common.read_SA_file(file_info['path'], previous_file=previous_filepath)
    pps_micros, adc_reading = sa_common.decode_SA_array(data_packets)
    unrolled_micros, rollovers = correct_micros(pps_micros, sample_rate)
    unrolled_micros = unrolled_micros + total_rollovers*u4max
    time_orig = np.array([file_info['dt']]).astype('datetime64[us]') + (unrolled_micros - unrolled_micros[0]).astype('timedelta64[us]')
    return unrolled_micros, adc_reading, time_orig, rollovers


def process_file_pair(file_info_1, file_info_2, output_dir, previous_filepath=None, SAMPLE_RATE=9600):
    # Read and decode raw data
    unrolled_micros_1, adc_reading_1, time_orig_1, rollovers_1 = read_single_file(file_info_1,
                                                                                  sample_rate=SAMPLE_RATE,
                                                                                  total_rollovers=0,
                                                                                  previous_filepath=previous_filepath)
    unrolled_micros_2, adc_reading_2, time_orig_2, _ = read_single_file(file_info_2,
                                                                                  sample_rate=SAMPLE_RATE,
                                                                                  total_rollovers=rollovers_1,
                                                                                  previous_filepath=file_info_1['path'])
    unrolled_micros = np.concatenate((unrolled_micros_1, unrolled_micros_2))
    adc_reading = np.concatenate((adc_reading_1, adc_reading_2))
    time_orig = np.concatenate((time_orig_1, time_orig_2))

    ds = xr.Dataset(pd.DataFrame(
            {'ADC':adc_reading,
             'pps_micro':unrolled_micros,
             'time_uncorrected':time_orig})).reset_index('dim_0').drop_vars('dim_0').rename_dims({'dim_0':'sample'})
    ds = interpolate_across_system_times(add_other_time_vars(ds))
    ds = ds.drop_vars(['dt_system', 'dt_adc', 'pps_micro', 'time_uncorrected'])
    ds = ds.sortby('time')
    sensor_num = file_info_1['sensor_num']
    sensor_lat = file_info_1['lat']
    sensor_lon = file_info_1['lon']
    sensor_alt = file_info_1['alt']
    sensor_relay = file_info_1['relay']
    this_dt = file_info_1['dt']
    fileoutname = f'SA{sensor_num}_{this_dt.strftime("%Y-%m-%d_%H-%M-%S")}.nc'
    fileoutpath = os.path.join(output_dir, fileoutname)
    if os.path.exists(fileoutpath):
        return
    ds = ds.assign_coords({'sensor_num' : np.array([sensor_num], dtype='i4')})        
    match sensor_relay:
        case 'a':
            sensor_relay = 0
        case 'b':
            sensor_relay = 1
        case 'c':
            sensor_relay = 2
        case _:
            raise ValueError(f'Unknown relay: {sensor_relay}')
    
    cpu_serial = hex(file_info_1['cpu_id'])
    ds = ds.assign(
        lat = ('sensor_num', np.array([sensor_lat], dtype='float32')),
        lon = ('sensor_num', np.array([sensor_lon], dtype='float32')),
        alt = ('sensor_num', np.array([sensor_alt], dtype='float32')),
        raspi_cpu_serial = ('sensor_num', np.array([cpu_serial],dtype=f'S{len(cpu_serial)}')),
        relay = ('sensor_num', np.array([sensor_relay], dtype='i4')),
        gps_err = ('sensor_num', np.array([file_info_1['gps_err']], dtype='float32')),
        geo_cal_scale = ('sensor_num', np.array([file_info_1['geo_cal_scale']], dtype='float32')),
        mass_cal_offset = ('sensor_num', np.array([file_info_1['mass_cal_offset']], dtype='float32')),
        resistor_ohms = ('sensor_num', np.array([file_info_1['resistor_ohms']], dtype='float32')),
        capacitor_farads = ('sensor_num', np.array([file_info_1['capacitor_farads']], dtype='float32')),
        RC_constant = ('sensor_num', np.array([file_info_1['RC_constant']], dtype='float32')),
        gain = ('sensor_num', np.array([file_info_1['gain']], dtype='float32')),
        static_cal_offset = ('sensor_num', np.array([file_info_1['static_cal_offset']], dtype='float32')),
        static_cal_scale = ('sensor_num', np.array([file_info_1['static_cal_scale']], dtype='float32'))
    )
    adc_var_long_name = 'Slow antenna ADC reading, raw'
    if ~np.isnan(file_info_1['mass_cal_offset']):
        adc_var_long_name = adc_var_long_name.replace('raw', 'noise floor adjusted to 0 by mass calibration')
        voltage_scaling = (5 / ((2**24) - 1))
        ds['ADC'] = (ds['ADC'].astype('float64') + file_info_1['mass_cal_offset']) * voltage_scaling
        voltage_offset = file_info_1['mass_cal_offset'] * voltage_scaling
        ds['ADC'].encoding['add_offset'] = voltage_offset
        ds['ADC'].encoding['scale_factor'] = voltage_scaling
        ds['ADC'].encoding['dtype'] = 'int32'
        ds['ADC'].encoding['_FillValue'] = -(2**31)
        ds['ADC'].attrs['units'] = 'volts'
    else:
        ds['ADC'].attrs['units'] = 'bits'
    ds['ADC'].attrs['long_name'] = adc_var_long_name
    ds['time'].attrs['long_name'] = 'Time of the ADC sample in UTC'
    ds['lat'].attrs['long_name'] = 'Latitude of the sensor'
    ds['lon'].attrs['long_name'] = 'Longitude of the sensor'
    ds['alt'].attrs['long_name'] = 'Altitude of the sensor'
    ds['raspi_cpu_serial'].attrs['long_name'] = 'Raspberry pi id number'
    ds['relay'].attrs['long_name'] = 'Active relay of the ADC, 0=a, 1=b, 2=c'
    ds['gps_err'].attrs['long_name'] = 'Time in seconds since last GPS fix'
    ds['geo_cal_scale'].attrs['long_name'] = 'Geometric calibration scale factor for this sensor'
    ds['mass_cal_offset'].attrs['long_name'] = 'Mass calibration offset for this sensor'
    ds['resistor_ohms'].attrs['long_name'] = 'Resistance in ohms of the RC circuit'
    ds['capacitor_farads'].attrs['long_name'] = 'Capacitance in farads of the RC circuit'
    ds['RC_constant'].attrs['long_name'] = 'Time constant in seconds of the RC circuit'
    ds['gain'].attrs['long_name'] = 'Gain of the channel in use relative to other channels'
    ds['sensor_num'].attrs['long_name'] = 'ADC board number'
    comp_ds = compress_all(ds.isel(sample=slice(0, len(adc_reading_1))))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(fileoutpath)
    comp_ds.to_netcdf(fileoutpath, engine='netcdf4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Slow Antenna .raw data')
    parser.add_argument('-i', '--input', nargs='+', help='Path or paths to slow antenna files to convert.')
    parser.add_argument('-o', '--output', help='Directory to save netCDF output files. If unspecified, will save in a "processed" subdirectory of the directory of the input files.')
    parser.add_argument('-m', '--hardware-metadata', default='./hardware.csv', help='Path to a CSV file containing hardware metadata history for the sensor network.')
    parser.add_argument('--sample-rate', type=int, default=9600, help='Sample rate of the ADC in samples/second. Default is 9600.')
    args = parser.parse_args()

        
    SAMPLE_RATE = args.sample_rate
    if args.input is not None:
        files_input = args.input
    else:
        files_input = sorted(glob('./input/**/*.raw', recursive=True))
    files = []
    filenames = []
    file_metadata_list = []
    file_sizes = []
    for f in files_input:
        file_size = os.path.getsize(f)
        if file_size == 0:
            print(f"Warning: File {f} is empty and will be skipped.")
            continue
        files.append(f)
        fn = os.path.basename(f)
        filenames.append(fn)
        file_metadata_list.append(sa_common.parse_filename(fn))
        file_sizes.append(file_size)
    hardware_df = pd.read_csv(args.hardware_metadata, parse_dates=['start_dt', 'end_dt'])
    for i, fm in enumerate(file_metadata_list):
        fm['filename'] = filenames[i]
        fm['path'] = files[i]
        fm['size_bytes'] = file_sizes[i]
        fm['cpu_id_hex'] = hex(fm['cpu_id'])
        hw_metadata = hardware_df[(hardware_df['cpu_serial'] == fm['cpu_id_hex']) & (hardware_df['start_dt'] <= fm['dt']) & (hardware_df['end_dt'] >= fm['dt'])]
        if len(hw_metadata) != 1:
            print(f"Warning: Could not find unique hardware metadata for file {filenames[i]}. Found {len(hw_metadata)} matches. Calibration metadata will not be applied for this file.")
        else:
            fm['sensor_num'] = hw_metadata['sensor_num'].values[0]
            fm['geo_cal_scale'] = hw_metadata['geo_cal_scale'].values[0]
            fm['mass_cal_offset'] = hw_metadata['mass_cal_offset'].values[0]
            relay_in_use = fm['relay']
            fm['resistor_ohms'] = hw_metadata[f'channel_{relay_in_use}_resistor_ohms'].values[0]
            fm['capacitor_farads'] = hw_metadata[f'channel_{relay_in_use}_capacitor_farads'].values[0]
            fm['RC_constant'] = hw_metadata[f'channel_{relay_in_use}_RC_const_seconds'].values[0]
            fm['gain'] = hw_metadata[f'channel_{relay_in_use}_gain'].values[0]
            fm['static_cal_offset'] = hw_metadata[f'static_cal_offset_channel_{relay_in_use}'].values[0]
            fm['static_cal_scale'] = hw_metadata[f'static_cal_scale_channel_{relay_in_use}'].values[0]
    file_metadata_df = pd.DataFrame(file_metadata_list)
    # sort files by datetime extracted from filename
    file_metadata_df = file_metadata_df.sort_values('dt').reset_index(drop=True)
    # 9 bytes per sample, 9600 samples/second, so compute the expected time of the next file, with a 10% tolerance
    bytes_per_message = np.zeros(len(file_metadata_df), dtype='i8')
    bytes_per_message[file_metadata_df['filename_spec'] <= 3] = 9
    expected_file_interval = (file_metadata_df['size_bytes'] / (bytes_per_message * SAMPLE_RATE)) * 1.1
    file_metadata_df['expected_next_dt'] = file_metadata_df['dt'] + pd.to_timedelta(expected_file_interval, unit='s')

    # Group by sensor number
    all_sensors_metadata_df = [group.reset_index(drop=True) for _, group in file_metadata_df.groupby('cpu_id_hex')]
    # setup dask for parallel processing
    n_cpus_to_use = os.cpu_count()//2
    cluster = LocalCluster(n_workers=n_cpus_to_use, threads_per_worker=1)
    client = cluster.get_client()
    all_res = []
    for this_sensor_metadata_df in all_sensors_metadata_df:
        # Group by sensor deployment, continouous periods of time where each 'next file' is recorded immediately after the previous (with some tolerance)
        deployment_idx = this_sensor_metadata_df['expected_next_dt'].shift(1) < this_sensor_metadata_df['dt']
        this_sensor_metadata_df['deployment_idx'] = deployment_idx.cumsum()
        all_deployments_metadata_df = [group.reset_index(drop=True) for _, group in this_sensor_metadata_df.groupby('deployment_idx')]
        # Process each deployment in parallel, with dask. Each deployment is a set of files that are continouous in time, so we can use the end of one file to correct the next.
        for this_deployment_metadata_df in all_deployments_metadata_df:
            # Find output dir if unspecified, then submit the processing job to dask.
            if args.output is None:
                output_dir = os.path.join(os.path.dirname(this_deployment_metadata_df.iloc[0]['path']), 'processed')
            else:
                output_dir = args.output
            # For the first file in the deployment, there is no previous file to use for correction, so set previous_filepath to None. 
            if len(this_deployment_metadata_df) > 1:
                all_res.append((this_deployment_metadata_df.iloc[0]['path'],
                                client.submit(process_file_pair, this_deployment_metadata_df.iloc[0],
                                this_deployment_metadata_df.iloc[1],
                                output_dir,
                                previous_filepath=None,
                                SAMPLE_RATE=SAMPLE_RATE)))
            else:
                print(f'Could not process file {this_deployment_metadata_df.iloc[0]["filename"]} because there is no previous file in the deployment to use for correction. Skipping this file.')
                continue
            # For subsequent files, use the previous file in the deployment.
            for i in range(2, len(this_deployment_metadata_df)):
                if args.output is None:
                    output_dir = os.path.join(os.path.dirname(this_deployment_metadata_df.iloc[i]['path']), 'processed')
                else:
                    output_dir = args.output
                all_res.append((this_deployment_metadata_df.iloc[i-1]['path'],
                                client.submit(process_file_pair, this_deployment_metadata_df.iloc[i-1],
                                            this_deployment_metadata_df.iloc[i],
                                            output_dir,
                                            previous_filepath=this_deployment_metadata_df.iloc[i-2]['path'],
                                SAMPLE_RATE=SAMPLE_RATE)))
    wait([future for _, future in all_res])
    failures = []
    for source_path, future in all_res:
        if future.status == 'error':
            failures.append((source_path, future.exception(), future.traceback()))
    client.close()
    cluster.close()

    n_ok = len(all_res) - len(failures)
    print(f'\nProcessed {n_ok} of {len(all_res)} file pairs successfully.')
    if failures:
        print(f'{len(failures)} file pairs failed:', file=sys.stderr)
        for source_path, exception, tb in failures:
            print(f'\n--- {source_path} ---', file=sys.stderr)
            traceback.print_exception(type(exception), exception, tb, file=sys.stderr)
        sys.exit(1)