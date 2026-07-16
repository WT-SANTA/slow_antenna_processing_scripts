#!/usr/bin/env python3
# Sorts TTU Slow Antenna network data and prunes unininteresting data by LMA
# Created 14 September 2025 by Sam Gardner <samuel.gardner@ttu.edu>

import argparse
from os import path
from pathlib import Path
from glob import glob
from sa_common import parse_filename
from datetime import datetime as dt, timedelta, UTC
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import h5py
from pyxlma import coords


def copy_file(old_path, new_path, reason=None, dry_run=True):
    if old_path == new_path:
        return
    if reason is not None:
        reason_str = f' ({reason})'
    else:
        reason_str = ''
    if dry_run:
        print(f'Would copy {old_path} to {new_path}{reason_str}')
    else:
        print(f'Copying {old_path} to {new_path}{reason_str}')
        Path(path.dirname(new_path)).mkdir(parents=True, exist_ok=True)
        if path.exists(new_path):
            with open(old_path, 'rb') as f:
                old_crc = crc32(f.read())
            with open(new_path, 'rb') as f:
                new_crc = crc32(f.read())
            if old_crc == new_crc:
                return
            else:
                unhandleable_file(f'File {new_path} already exists and has different contents than {old_path}. Cannot copy {old_path} to {new_path} without overwriting a different file. Please resolve this conflict manually.', dry_run=dry_run)
        copyfile(old_path, new_path)

def unhandleable_file(message, dry_run=True):
    if dry_run:
        print(f'Would raise error: {message}')
    else:
        raise ValueError(message)

def parse_all_filenames(raw_paths):
    raw_files = [f.name for f in raw_paths]
    filenames_parsed = []
    for i, f in enumerate(raw_files):
        try:
            parsed = parse_filename(f)
            parsed['raw_path'] = raw_paths[i]
            filenames_parsed.append(parsed)
        except ValueError as e:
            if 'does not match any known filename specifications' in str(e) and len(f.split('_')) == 6:
                rawfile_split = f.replace('.raw', '').split('_')
                this_date = dt.strptime(rawfile_split[0]+rawfile_split[1]+rawfile_split[2], '%Y%m%d%H%M%S%f')
                this_cpu = int(rawfile_split[4], 16)
                this_relay = rawfile_split[5]
                this_lat = np.nan
                this_lon = np.nan
                this_alt = np.nan
                this_gps_err = 0
                parsed = {
                    'filename_spec': 3,
                    'dt': this_date,
                    'relay': this_relay,
                    'lon': this_lon,
                    'lat': this_lat,
                    'alt': this_alt,
                    'gps_err': this_gps_err,
                    'cpu_id': this_cpu,
                    'raw_path': raw_paths[i]
                }
                filenames_parsed.append(parsed)
            else:
                unhandleable_file(f'Filename {f} could not be parsed: {e}', dry_run=args.dry_run)
    # recover sensor dir info
    filenames_parsed = pd.DataFrame(filenames_parsed)
    return filenames_parsed

def associate_sensor_nums(filenames_parsed):
    filenames_parsed['sensor_dir'] = ''
    filenames_parsed['sensor_num'] = 0
    for i, rp in enumerate(filenames_parsed['raw_path']):
        parts = rp.parts
        psbl_sensor_paths = [p for p in parts if 'sensor_' in p]
        if len(psbl_sensor_paths) != 1:
            unhandleable_file(f'Could not find sensor directory in path {rp}', dry_run=args.dry_run)
        else:
            filenames_parsed.at[i, 'sensor_dir'] = psbl_sensor_paths[0]
            try:
                filenames_parsed.at[i, 'sensor_num'] = int(psbl_sensor_paths[0].replace('sensor_', ''))
            except ValueError:
                unhandleable_file(f'Could not parse sensor number from directory {psbl_sensor_paths[0]} in path {rp}', dry_run=args.dry_run)
    return filenames_parsed

def upgrade_old_filenames(filenames_parsed, history_df):
    old_data_df = filenames_parsed.loc[filenames_parsed['filename_spec'] < 3]
    for i, parsed in old_data_df.iterrows():
        old_raw_path = parsed['raw_path']
        this_dt = parsed["dt"]
        history_row = history_df.loc[(history_df['start_date'] <= this_dt) & (this_dt <= history_df['end_date']) & (history_df['sensor_num'] == parsed['sensor_num'])]
        if len(history_row) != 1:
            unhandleable_file(f'File {old_raw_path} has datetime {this_dt} and sensor number {parsed["sensor_num"]} that falls within {len(history_row)} rows of the history file. Cannot determine how to rename and sort this file without a unique matching row in the history file.', dry_run=args.dry_run)
            continue
        if history_row['needs_utc_correction'].values[0]:
            this_dt_aware = this_dt.replace(tzinfo=ZoneInfo('America/Chicago'))
            this_dt_utc = this_dt_aware.astimezone(UTC)
            this_dt = this_dt_utc.replace(tzinfo=None)
        this_lat = history_row['lat'].values[0]
        this_lon = history_row['lon'].values[0]
        this_alt = history_row['alt'].values[0]
        this_gps_err = 0
        this_cpu_id = int(history_row['cpu_id'].values[0], 16)
        this_relay = parsed['relay']
        if parsed['filename_spec'] == 2 or ~np.isnan(this_relay):
            pass
        else:
            this_relay = history_row['relay'].values[0]
        old_data_df.at[i, 'lat'] = this_lat
        old_data_df.at[i, 'lon'] = this_lon
        old_data_df.at[i, 'alt'] = this_alt
        old_data_df.at[i, 'gps_err'] = this_gps_err
        old_data_df.at[i, 'cpu_id'] = this_cpu_id
        old_data_df.at[i, 'relay'] = this_relay
    filenames_parsed.update(old_data_df)
    return filenames_parsed


def fix_issue_three(filenames_parsed):
    issue_3_files = filenames_parsed.loc[(filenames_parsed['filename_spec'] == 3) & (filenames_parsed['gps_err'] == 0) & (filenames_parsed['lat'].isna()) & (filenames_parsed['lon'].isna()) & (filenames_parsed['alt'].isna())]
    for i, row in issue_3_files.iterrows():
        nearby_files = filenames_parsed.loc[(filenames_parsed['filename_spec'] == 3) & 
                                            (filenames_parsed['dt'] >= row['dt'] - timedelta(minutes=3)) &
                                            (filenames_parsed['dt'] <= row['dt'] + timedelta(minutes=3)) &
                                            (filenames_parsed['sensor_num'] == row['sensor_num'])]
        if len(nearby_files) == 0:
            continue
        nearby_lons = np.nanmean(nearby_files['lon'])
        nearby_lats = np.nanmean(nearby_files['lat'])
        nearby_alts = np.nanmean(nearby_files['alt'])
        filenames_parsed.at[i, 'lon'] = nearby_lons
        filenames_parsed.at[i, 'lat'] = nearby_lats
        filenames_parsed.at[i, 'alt'] = nearby_alts
    return filenames_parsed


def create_cpu_serial_log(filenames_parsed):
    error_msgs = []
    unique_hex_serials = np.unique([hex(int(this_parsed)) for this_parsed in filenames_parsed['cpu_id'].values if ~np.isnan(this_parsed)])
    all_dates = np.unique(filenames_parsed['dt'].dt.normalize())
    cpu_serial_df = pd.DataFrame(index=pd.Index(unique_hex_serials, name='CPU Serial'), columns=[d.strftime('%Y%m%d') for d in np.sort(np.unique(all_dates)).astype('datetime64[D]').astype('O')], dtype='float')
    for i, (sensor_num, cpu_serial, date) in enumerate(zip(filenames_parsed['sensor_num'], filenames_parsed['cpu_id'], filenames_parsed['dt'].dt.strftime('%Y%m%d'))):
        if np.isnan(cpu_serial):
            continue
        hex_serial = hex(int(cpu_serial))
        if np.isnan(cpu_serial_df.at[hex_serial, date]) or cpu_serial_df.at[hex_serial, date] == sensor_num:
            cpu_serial_df.at[hex_serial, date] = sensor_num
        else:
            error_msgs.append(f'CPU serial {hex_serial} is associated with multiple sensor numbers on {date}: {cpu_serial_df.at[hex_serial, date]} and {sensor_num}.')
    if len(error_msgs) > 0:
        print('\n'.join(np.unique(error_msgs).tolist()))
    cpu_serial_df = cpu_serial_df.fillna(0.0)
    cpu_serial_df = cpu_serial_df.astype(int)
    cpu_serial_df.to_csv(args.cpu_serial_log, index=True)


def sort_files(filenames_parsed, archive_root):
    files_to_move = filenames_parsed.loc[filenames_parsed['keep']]
    for i, parsed in files_to_move.iterrows():
        if np.isnan(parsed['cpu_id']):
            continue
        old_raw_path = parsed['raw_path']
        this_lat = parsed["lat"] if ~np.isnan(parsed["lat"]) else 'NO'
        this_lon = parsed["lon"] if ~np.isnan(parsed["lon"]) else 'FIX'
        this_alt = parsed["alt"] if ~np.isnan(parsed["alt"]) else '2DONLY'
        this_gps_err = parsed["gps_err"] if ~np.isnan(parsed["gps_err"]) else 'NaT'
        this_cpu_id = hex(int(parsed["cpu_id"])).replace("0x", "") if ~np.isnan(parsed["cpu_id"]) else '0'
        this_relay = parsed["relay"] if parsed["relay"] in ['a', 'b', 'c'] else ''

        new_path = path.join(archive_root, f'{parsed["dt"].strftime("%Y%m%d")}', f'sensor_{str(int(parsed["sensor_num"])).zfill(2)}', 
                                f'{parsed["dt"].strftime("%Y%m%d_%H%M%S_%f")}_{this_lat}_{this_lon}_{this_alt}_{this_gps_err}_{this_cpu_id}_{this_relay}.raw')
        copy_file(old_raw_path, new_path, dry_run=args.dry_run)
        print(f'Processed file {i+1} of {files_to_move.shape[0]}')


def move_log_files(unsorted_files_dir, archive_root):
    log_files = glob(path.join(unsorted_files_dir, '**', 'SA_log.out'), recursive=True)
    log_files += glob(path.join(unsorted_files_dir, '**', 'cronjobs_help.sh'), recursive=True)
    log_files += glob(path.join(unsorted_files_dir, '**', 'cronlog.txt'), recursive=True)
    log_files += glob(path.join(unsorted_files_dir, '**', 'cronlog.out'), recursive=True)
    log_files += glob(path.join(unsorted_files_dir, '**', 'data_collect.py'), recursive=True)
    log_files += glob(path.join(unsorted_files_dir, '**', 'temperature_log.csv'), recursive=True)
    for log_file in log_files:
        new_path = log_file.replace(unsorted_files_dir, archive_root)
        if path.exists(path.dirname(new_path)):
            copy_file(log_file, new_path, reason='log file', dry_run=args.dry_run)
        else:
            unhandleable_file(f'Could not find destination directory for log file {log_file}. Expected to find {path.dirname(new_path)}. Please resolve this issue manually.', dry_run=args.dry_run)


def filter_lma(filenames_parsed, lma_data_path):
    geosys = coords.GeographicSystem()
    filenames_parsed['filtered_by_lma'] = False
    files_that_can_be_pruned = filenames_parsed.loc[~filenames_parsed['dt'].isna() & ~filenames_parsed['lat'].isna() & ~filenames_parsed['lon'].isna()]
    days_in_data = files_that_can_be_pruned['dt'].dt.normalize().dropna().unique()
    lma_lat, lma_lon, lma_alt = None, None, None
    for this_date in sorted(pd.to_datetime(days_in_data)):
        df_this_day = files_that_can_be_pruned.loc[(files_that_can_be_pruned['dt'] >= this_date) & (files_that_can_be_pruned['dt'] < this_date + timedelta(days=1))]
        first_time_this_day = df_this_day['dt'].values.min().astype('datetime64[s]')
        last_time_this_day = df_this_day['dt'].values.max().astype('datetime64[s]')
        # Find LMA files that cover this time range
        lma_pattern = path.join(lma_data_path, this_date.strftime('%Y/%b/%d/*_%y%m%d_*_0600.dat.flash.h5'))
        lma_file_paths = sorted(glob(lma_pattern))
        tomorrow = this_date + timedelta(days=1)
        lma_file_paths += sorted(glob(path.join(lma_data_path, tomorrow.strftime('%Y/%b/%d/*_%y%m%d_000000_0600.dat.flash.h5'))))
        lma_file_times = np.array([dt.strptime(''.join(path.basename(lf).replace('_2011_', '_').split('_')[1:3]), '%y%m%d%H%M%S') for lf in lma_file_paths], dtype='datetime64[s]')
        files_to_read_mask = (lma_file_times >= first_time_this_day) & (lma_file_times <= last_time_this_day)
        lma_file_times = lma_file_times[files_to_read_mask].astype('datetime64[s]').astype(dt)
        lma_file_paths = np.array(lma_file_paths)[files_to_read_mask].tolist()
        if len(lma_file_paths) == 0:
            continue
        if lma_lat is None or lma_lon is None or lma_alt is None:
            first_ds = h5py.File(lma_file_paths[0], 'r')
            lma_header = first_ds['events'][lma_file_times[0].strftime('LMA_%y%m%d_%H%M%S_600')].attrs['header'].decode()
            lma_header_split = lma_header.split('\n')
            lma_header_coords = [l for l in lma_header_split if 'Coordinate center (lat,lon,alt):' in l][0]
            lma_lat, lma_lon, lma_alt = [float(s) for s in lma_header_coords.replace('Coordinate center (lat,lon,alt):', '').strip().split()]
            first_ds.close()
        lma_X, lma_Y, lma_Z = geosys.toECEF(lma_lon, lma_lat, lma_alt)
        sensor_X, sensor_Y, sensor_Z = geosys.toECEF(df_this_day['lon'].values, df_this_day['lat'].values, df_this_day['alt'].values)
        sensor_distances = ((sensor_X - lma_X)**2 + (sensor_Y - lma_Y)**2 + (sensor_Z - lma_Z)**2)**0.5
        if np.any(sensor_distances > 100e3):
            print(f'WARNING: Sensor more than 100 km from the LMA center for date {this_date.strftime("%Y-%m-%d")}.')
            continue
        else:
            print(f'Max sensor distance from LMA center for date {this_date.strftime("%Y-%m-%d")}: {np.max(sensor_distances):.2f} m.')
        if len(lma_file_paths) > 0:
            flash_df = pd.DataFrame()
            for i, this_path in enumerate(lma_file_paths):
                ds = h5py.File(this_path, 'r')
                this_flash_df = pd.DataFrame(ds['flashes'][lma_file_times[i].strftime('LMA_%y%m%d_%H%M%S_600')][:])
                this_flash_df['flash_dt'] = pd.Timestamp(lma_file_times[i].replace(hour=0, minute=0, second=0, microsecond=0)) + pd.to_timedelta(this_flash_df['start'], unit='s')
                flash_df = pd.concat([flash_df, this_flash_df], ignore_index=True).reset_index(drop=True)
                ds.close()
            if flash_df.shape[0] > 0:
                flash_df = flash_df[flash_df['n_points'] >= 20] # only consider flashes with 20 or more points
                sensor_X, sensor_Y, sensor_Z = geosys.toECEF(df_this_day['lon'].values, df_this_day['lat'].values, np.zeros(df_this_day['lon'].shape))
                flash_X, flash_Y, flash_Z = geosys.toECEF(flash_df['ctr_lon'].values, flash_df['ctr_lat'].values, np.zeros(flash_df['ctr_lon'].shape))
                distances = ((sensor_X.reshape((-1, 1)) - flash_X.reshape((1, -1)))**2
                        + (sensor_Y.reshape((-1, 1)) - flash_Y.reshape((1, -1)))**2
                        + (sensor_Z.reshape((-1, 1)) - flash_Z.reshape((1, -1)))**2)**0.5
                distances_thresholded = distances <= 100e3 # 100 km
                times_differences = np.abs((df_this_day['dt'].values.reshape((-1, 1)) - flash_df['flash_dt'].values.reshape((1, -1))).astype('timedelta64[s]').astype(float))
                times_differences_thresholded = times_differences <= 1800 # 30 minutes
                flashes_nearby = np.any(distances_thresholded & times_differences_thresholded, axis=1)
                paths_to_rm = df_this_day.loc[~flashes_nearby, 'raw_path']
                filenames_parsed.loc[filenames_parsed['raw_path'].isin(paths_to_rm), 'filtered_by_lma'] = True
            else:
                filenames_parsed.loc[filenames_parsed['dt'].dt.normalize() == this_date, 'filtered_by_lma'] = True
    return filenames_parsed


def filter_triggers(filenames_parsed):
    print('TODO: PUT TRIGGERS HERE!')
    filenames_parsed['filtered_by_trigger'] = False
    return filenames_parsed


def filter_empty(filenames_parsed):
    filenames_parsed['filtered_by_empty'] = False
    for i, row in filenames_parsed.iterrows():
        if path.getsize(row['raw_path']) == 0:
            filenames_parsed.at[i, 'filtered_by_empty'] = True
    return filenames_parsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slow antenna archive pruning and sorting.')
    parser.add_argument('--unsorted-files', '-u', type=str, required=True, default='/mnt/reservoir/SA_DATA_UNSORTED/SA_DATA/', help='Top level directory of unsorted slow antenna data.')
    parser.add_argument('--archive-root', '-r', type=str, default='/mnt/reservoir/SA_DATA/', help='Top level directory of slow antenna data archive.')
    parser.add_argument('--lma-data', '-l', type=str, default=None, help='Directory containing LMA data for correlation with slow antenna data. On rime as of Oct. 2025, this is "/archive/lmaclimo/h5_files/". Leave unspecified to skip pruning.')
    parser.add_argument('--trigger', '-t', action='store_true', default=False, help='Whether to prune based on the presence of lightning impulses in the slow antenna data.')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Log actions without making any changes.')
    parser.add_argument('--cpu-serial-log', '-c', type=str, default=None, help='Path to a CSV file logging CPU serial numbers and their associated sensor numbers and deployment dates. Leave unspecified to skip.')
    parser.add_argument('--history-file', type=str, help='Path to a CSV file logging the history of previous deployments that collected \'old old\' or \'old\' data. Files with data collection dates that fall within the date ranges of these deployments will be renamed to the current filename specification and sorted into the archive according to their collection date and sensor number.')
    parser.add_argument('--debug-dataframe', '-d', type=str, default=None, help='Path to write the dataframe containing all parsed filename information and pruning decisions to a CSV for debugging purposes. Leave unspecified to skip.')

    args = parser.parse_args()
    if not args.dry_run:
        print('----> FILESYSTEM CHANGES CAN BE PERFORMED <----')
        from shutil import copyfile
        from time import sleep
        from zlib import crc32
        print('----> Press CTRL + C in the next 5 seconds to abort! <----')
        sleep(5)
    # Read history, if provided
    if args.history_file is not None:
        history_df = pd.read_csv(args.history_file, parse_dates=['start_date', 'end_date'])
    else:
        history_df = pd.DataFrame(columns=['sensor_num', 'start_date', 'end_date', 'lat', 'lon', 'alt', 'cpu_id', 'relay', 'needs_utc_correction'])
    # Get full paths to all raw files
    raw_files = glob(path.join(args.unsorted_files, '**', '*.raw'), recursive=True)
    raw_paths = [Path(f) for f in raw_files]
    # Do all the processing things
    filenames_parsed = parse_all_filenames(raw_paths)
    filenames_parsed = associate_sensor_nums(filenames_parsed)
    if args.cpu_serial_log is not None:
        create_cpu_serial_log(filenames_parsed)
    filenames_parsed = upgrade_old_filenames(filenames_parsed, history_df)
    filenames_parsed = fix_issue_three(filenames_parsed)
    # Pruning
    filenames_parsed['keep'] = True
    filenames_parsed = filter_empty(filenames_parsed)
    filenames_parsed['keep'] = filenames_parsed['keep'] & ~filenames_parsed['filtered_by_empty']
    if args.lma_data is not None:
        filenames_parsed = filter_lma(filenames_parsed, args.lma_data)
        filenames_parsed['keep'] = filenames_parsed['keep'] & ~filenames_parsed['filtered_by_lma']
    if args.trigger:
        filenames_parsed = filter_triggers(filenames_parsed)
        filenames_parsed['keep'] = filenames_parsed['keep'] & ~filenames_parsed['filtered_by_trigger']
    filenames_parsed = filenames_parsed.sort_values('dt').reset_index(drop=True)
    sort_files(filenames_parsed, args.archive_root)
    if args.debug_dataframe is not None:
        if args.debug_dataframe.endswith('.csv'):
            filenames_parsed.to_csv(args.debug_dataframe, index=False)
        elif args.debug_dataframe.endswith('.parquet'):
            filenames_parsed.to_parquet(args.debug_dataframe, index=False)
    move_log_files(args.unsorted_files, args.archive_root)
