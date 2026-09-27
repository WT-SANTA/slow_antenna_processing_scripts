#!/usr/bin/env python3
# Transfer data from slow antenna and organize
# Created 16 August 2024 by Sam Gardner <samuel.gardner@ttu.edu>


from paramiko import SSHClient, AutoAddPolicy
from os import listdir, path, rename
from shutil import rmtree
import subprocess
from datetime import datetime as dt, UTC
from pathlib import Path
import pandas as pd
import numpy as np
from timeoutcontext import timeout
from alive_progress import alive_bar
import re

rsync_started = False
rsync_finished = False

kelcy_drive_path = '/media/lightning/Extreme_SSD/'


def sort_file_dates(sensor_ingest_dir):
    new_files = sorted(listdir(sensor_ingest_dir))
    new_dates = []
    for new_file in new_files:
        new_date_str = new_file[:8]
        new_dates.append(new_date_str)
    unique_dates = np.unique(new_dates)
    new_file_paths = [path.join(sensor_ingest_dir, f) for f in new_files]
    for date_str in unique_dates:
        date_dir = sensor_ingest_dir.replace('unsorted_dates', date_str)
        Path(date_dir).mkdir(parents=True, exist_ok=True)
        for file in new_file_paths:
            if path.basename(file).startswith(date_str):
                rename(file, file.replace('unsorted_dates', date_str))

    if len(listdir(sensor_ingest_dir)) == 0:
        rmtree(sensor_ingest_dir)
    else:
        print(listdir(sensor_ingest_dir))
        raise ValueError('Failed to empty unsorted_dates directory')
    return unique_dates



def process_rsync_output(line, bar):
    line = line.decode()
    if 'rsync warning' in line:
        print(line)
    if '%' in line:
        pct = float(line.split()[1].replace('%', ''))/100
        bar(pct)
    elif 'speedup is' in line:
        bar(1)
    return -1


def run_rsync_with_progress(source_path, dest_path, should_SSH=True):
    rsync_cmd = ['rsync', '-uvr', '--no-i-r', '--info=progress2', source_path, dest_path]
    if should_SSH:
        prepend = ['sshpass', '-p', 'raspberry']
        prepend.extend(rsync_cmd)
        rsync_cmd = prepend
    
    rsync_task = subprocess.Popen(rsync_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    with alive_bar(manual=True, spinner='arrow', title='Transferring Data...', elapsed='total {elapsed}', stats='(ETA: {eta})') as bar:
        for line in iter(rsync_task.stdout.readline, b''):
            process_rsync_output(line, bar)

    rsync_task.stdout.close()
    rsync_task.wait()
    errors = rsync_task.stderr.read().decode().split('\n')
    errors = [e for e in errors if "Permanently added 'raspberrypi.local'" not in e]
    errors = [e for e in errors if e != '']
    if errors:
        print('##################################################')
        print('WARNING! Errors detected in rsync process!')
        print('You may wish to CTRL+C to exit and try again before proceeding')
        print('')
        print(errors)
        print('')
        print('##################################################')
        input('Press RETURN to proceed, CTRL+C to exit...')
    return rsync_task


if __name__ == '__main__':
    _ = input('Please connect power and network to the slow antennta and press RETURN when connected...')
    
    with SSHClient() as client:
        client.set_missing_host_key_policy(AutoAddPolicy())
        client.connect('raspberrypi.local', username='pi', password='raspberry')
        stdin, stdout, stderr = client.exec_command('cat /proc/cpuinfo')
        stdout_list = stdout.readlines()
    cpu_serial = None
    for i in range(len(stdout_list)):
        line = stdout_list[i]
        if 'Serial' in line:
            cpu_serial = line[-9:-1]
    stdin.close()
    try:
        raise OSError('(Sharepoint finding disabled manually...)')
        #with timeout(10):
        #    files_in_sharepoint = listdir('/home/lightning/Desktop/SharePoint_mirror')
    except TimeoutError as e:
        print(e)
        files_in_sharepoint = []
    except OSError as e:
        print('######################')
        print('rclone mount is unavailable! You may wish to CTRL+C to exit, type `sudo systemctl restart rclone_mount`, and try again.')
        print('(Also, check your internet connection...)')
        print(e)
        print('######################')
        files_in_sharepoint = []
    if 'Inventory.xlsx' not in files_in_sharepoint:
        if len(files_in_sharepoint) == 0:
            print('Sharepoint mount appears to be unavailable...')
        else:
            print('Cannot find Invensory.xlsx in SharePoint...')
        pi_num = int(input('Which slow antenna system # is this? '))
    else:
        try:
            inv = pd.read_excel('/home/lightning/Desktop/SharePoint_mirror/Inventory.xlsx', sheet_name='Raspberry Pi CPU Serial Numbers', header=None, names=['Pi#', 'CPU'])
            pi_num = inv[inv['CPU'] == cpu_serial]['Pi#'].values[0]
            inv2 = pd.read_excel('/home/lightning/Desktop/SharePoint_mirror/Inventory.xlsx', sheet_name='Slow Antenna Systems')
            pi_num = int(re.findall(r'\d+', inv2[inv2['Raspberry Pi 4'] == pi_num]['Slow Antenna'].values[0])[0])
            is_correct = input(f'This appears to be system #{pi_num}. Is this correct? [Y/n]: ')
            if is_correct.lower() == 'y' or is_correct == '':
                pass
            else:
                pi_num = int(input('Please enter correct system number: '))
        except Exception as e:
            print(e)
            pi_num = int(input('Which slow antenna system # is this? '))
    ingest_date = dt.now(UTC)
    ingest_dir = f'/home/lightning/Desktop/ingest/unsorted_dates/'
    Path(ingest_dir).mkdir(parents=True, exist_ok=True)
    sensor_dir = path.join(ingest_dir, f'sensor_{str(pi_num).zfill(2)}')
    rsync_task = run_rsync_with_progress('pi@raspberrypi.local:/home/pi/Desktop/DATA/.', sensor_dir, should_SSH=True)
    new_dates = sort_file_dates(sensor_dir)
    if len(listdir(ingest_dir)) == 0:
        rmtree(ingest_dir)
    print('Successfully wrote data to: ')
    [print(f'   - {ingest_dir.replace("unsorted_dates", new_date)}') for new_date in new_dates]
    should_purge = input('Would you like to purge all raw files from the slow antenna data directory? [Y/n]: ')
    if should_purge.lower() == 'y' or should_purge == '':
        with SSHClient() as client:
            client.set_missing_host_key_policy(AutoAddPolicy())
            client.connect('raspberrypi.local', username='pi', password='raspberry')
            stdin, stdout, stderr = client.exec_command('rm -rf /home/pi/Desktop/DATA/*.raw*')
            print(stderr.read().decode())
            print('Done!')
    print('You may now disconnect and power off the slow antenna.')
