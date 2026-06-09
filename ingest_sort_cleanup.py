#!/usr/bin/env python3
# Sorts and filters incoming WT-SANTA data for processing and archival storage.
# Created 2 April 2026 by Sam Garnder <samuel.gardner@ttu.edu>

import argparse
from os import listdir, path, rename
from glob import glob
import pandas as pd
from pathlib import Path
from time import sleep
from sa_common import parse_filename
from datetime import datetime as dt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sort and filter incoming WT-SANTA data for processing and archival storage.')
    parser.add_argument('-i', '--ingest_dir', type=str, help='Path to the directory containing the ingested data files.', default='/mnt/reservoir/ingest_data/')
    parser.add_argument('-u', '--unfiltered_output_dir', type=str, help='Path to the directory where sorted, unfiltered files will be stored.', default='/mnt/reservoir/SA_DATA_UNFILTERED/SA_DATA/')
    parser.add_argument('-f', '--filtered_output_dir', type=str, help='Path to the directory where sorted, filtered files will be stored.', default='/mnt/reservoir/SA_DATA/')
    parser.add_argument('-m', '--metadata_csv', type=str, help='Path to the CSV file containing WT-SANTA hardware metadata', default='/mnt/reservoir/slow_antenna_processing_scripts/hardware.csv')
    args = parser.parse_args()

    hardware_info = pd.read_csv(args.metadata_csv, parse_dates=['start_dt', 'end_dt'])
    hardware_info['cpu_serial_10'] = hardware_info['cpu_serial'].apply(int, base=16)
    error_files = []

    about_to_exit_flag = False
    while True:
        paths_to_process = [p.absolute() for p in Path(args.ingest_dir).rglob('*.raw')]
        paths_to_process = [p for p in paths_to_process if p.name not in error_files]
        if len(paths_to_process) == 0:
            if about_to_exit_flag:
                if len(error_files) > 0:
                    print(f'The following files were not sorted due to errors:\n{"\n".join(error_files)}')
                else:
                    print('All files sorted!')
                break
            about_to_exit_flag = True
            print('I think we\'re done here, waiting 30 seconds before exiting...')
            sleep(30)
            continue
        about_to_exit_flag = False
        for i, path_to_process in enumerate(paths_to_process):
            file_to_process = path_to_process.name
            file_info_dict = parse_filename(file_to_process)
            if pd.isnull(file_info_dict['cpu_id']):
                error_files.append(file_to_process)
                continue
            if type(file_info_dict['dt']) != dt:
                error_files.append(file_to_process)
                continue
            sensor_info = hardware_info.loc[hardware_info['cpu_serial_10'] == file_info_dict['cpu_id']]
            sensor_info = sensor_info.loc[hardware_info['start_dt'] <= file_info_dict['dt']].loc[hardware_info['end_dt'] >= file_info_dict['dt']]
            if len(sensor_info) == 0:
                error_files.append(file_to_process)
                continue
            sensor_num = sensor_info.iloc[-1]['sensor_num']
            output_dir = path.join(args.unfiltered_output_dir, file_info_dict['dt'].strftime('%Y%m%d'), f'sensor_{str(sensor_num).zfill(2)}')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = path.join(output_dir, file_to_process)
            rename(path_to_process, output_path)