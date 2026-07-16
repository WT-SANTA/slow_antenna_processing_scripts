#!/usr/bin/env python3
# Common function "library" for slow antenna processing scripts
# Created 29 September 2025 by Sam Gardner <samuel.gardner@ttu.edu>

import numpy as np
from datetime import datetime as dt

def parse_filename(filename):
    out_dict = {
        'filename_spec': np.nan,
        'dt': np.nan,
        'relay': np.nan,
        'lon': np.nan,
        'lat': np.nan,
        'alt': np.nan,
        'gps_err': 0,
        'cpu_id': np.nan
    }
    rawfile_split = filename.replace('.raw', '').split('_')
    match len(rawfile_split):
        case 2:
            # this is an 'old old' file type
            out_dict['filename_spec'] = 1
            out_dict['dt'] = dt.strptime(filename, '%Y%m%d%H%M%S_%f.raw')
        case 3:
            # this is an 'old' file type
            out_dict['filename_spec'] = 2
            out_dict['dt'] = dt.strptime(rawfile_split[0]+rawfile_split[1], '%Y%m%d%H%M%S%f')
            out_dict['relay'] = rawfile_split[2]
        case 6:
            # This is a current file with no GPS... see https://github.com/wx4stg/Bruning_Slow_Antenna_Software/issues/3
            out_dict['filename_spec'] = 3
            out_dict['dt'] = dt.strptime(rawfile_split[0]+rawfile_split[1]+rawfile_split[2], '%Y%m%d%H%M%S%f')
            out_dict['lon'] = np.nan
            out_dict['lat'] = np.nan
            out_dict['alt'] = np.nan
            out_dict['gps_err'] = float(rawfile_split[3]) if ~np.isnan(out_dict['lon']) else 0
            out_dict['cpu_id'] = int(rawfile_split[4], 16)
            out_dict['relay'] = rawfile_split[5]
        case 9:
            # this is a current filename
            out_dict['filename_spec'] = 3
            out_dict['dt'] = dt.strptime(rawfile_split[0]+rawfile_split[1]+rawfile_split[2], '%Y%m%d%H%M%S%f')
            out_dict['lat'] = np.nan if rawfile_split[3] == 'NO' else float(rawfile_split[3])
            out_dict['lon'] = np.nan if rawfile_split[4] == 'FIX' else float(rawfile_split[4])
            out_dict['alt'] = np.nan if rawfile_split[5] == '2Donly' else float(rawfile_split[5])
            out_dict['gps_err'] = float(rawfile_split[6]) if ~np.isnan(out_dict['lon']) else 0
            out_dict['cpu_id'] = int(rawfile_split[7], 16)
            out_dict['relay'] = rawfile_split[8]
        case _:
            raise ValueError(f'Filename {filename} does not match any known filename specifications.')
    return out_dict



def rotate_SA_array(this_bytes, packet_length=9, previous_file=None):
    j = 0
    for j, byte in enumerate(this_bytes):
        # Find the first byte that is the header of the an ADC packet
        # The ADC packet has a header of 'BE' and a footer of 'EF' (190 = BE, 239 = EF)
        # So look for a footer immediately followed by a header,
        # followed by another footer/header pair 9 bytes later
        if byte == 190 and this_bytes[j+packet_length-1] == 239 and this_bytes[j+packet_length] == 190:
            length_to_include = ((this_bytes[j:].shape[0]) // packet_length) * packet_length
            adc_packets = this_bytes[j:length_to_include+j].reshape(-1, packet_length)
            # Now check all of the first and last columns to make sure they are all headers and footers respectively
            if np.all(adc_packets[:, 0] == 190) and np.all(adc_packets[:, -1] == 239):
                break
        if j >= packet_length:
            raise ValueError('Could not find first ADC packet in file.')
    if previous_file and j > 0:
        with open(previous_file, 'rb') as f:
            f.seek(-packet_length+j, 2)
            last_ba = f.read()
        last_bytes = np.frombuffer(last_ba, dtype=np.uint8)
        first_packet_recovered = np.append(last_bytes, this_bytes[:j])
        first_packet_recovered = first_packet_recovered.reshape(-1, packet_length)
        if first_packet_recovered[0, 0] == 190 and first_packet_recovered[0, -1] == 239:
            adc_packets = np.append(first_packet_recovered, adc_packets, axis=0)
        else:
            print('First packet not recovered')
    return adc_packets


def read_SA_file(file_to_read, packet_length=9, previous_file=None):
    with open(file_to_read, 'rb') as f:
        ba = f.read()
    this_bytes = np.frombuffer(ba, dtype=np.uint8)
    return rotate_SA_array(this_bytes, packet_length=packet_length, previous_file=previous_file)


def decode_SA_array(data_array):
    # Extract the ADC local clock and convert the hexadecimal value to base 10
    # ADC clock is sent as 4 bytes, least significant byte first, so multiply by 256^0, 256^1, 256^2, 256^3
    adc_pps_micros = np.sum(data_array[:, 4:8] * (256 ** np.arange(4)), axis=1)
    # Extract the ADC reading and convert to decimal.
    # The ADC reading is sent as 3 bytes, most significant byte first, so multiply by 256^2, 256^1, 256^0
    adc_reading_dec = np.sum(data_array[:, 1:4] * np.flip(256 ** np.arange(3)), axis=1)
    # The ADC reading is sent as an unsigned 24-bit integer, so we need to convert it to a signed 23-bit integer
    # If the ADC reading is greater than 2^23 - 1, then it is a negative number
    adc_reading_overflow_mask = adc_reading_dec > (2**23 - 1)
    adc_reading = adc_reading_dec - adc_reading_overflow_mask.astype(int) * (2**24)
    return adc_pps_micros, adc_reading.astype(np.int32)
