#!/usr/bin/env python3
# Common function "library" for slow antenna processing scripts
# Created 29 September 2025 by Sam Gardner <samuel.gardner@ttu.edu>

import numpy as np
from datetime import datetime as dt
import gzip

class open_gzip_or_dat:
    """Helper class open a file with gzip if necessary or as binary if already decompressed.

    Use as a context manager in the same way as `with open(filename) as f:`. If the filename ends with
    '.gz', the file will be decompressed with gzip. Otherwise, the file will be opened as binary.
    """
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        if self.filename.endswith('.gz'):
            self.file = gzip.open(self.filename)
        else:
            self.file = open(self.filename, 'rb')
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

def parse_filename(filename):
    out_dict = {
        'filename_spec': np.nan,
        'dt': np.nan,
        'relay': np.nan,
        'lon': np.nan,
        'lat': np.nan,
        'alt': np.nan,
        'gps_err': 0,
        'cpu_id': np.nan,
        'gzipped' : filename.endswith('.gz'),
        'embedded_gps_pps' : False
    }
    rawfile_split = filename.replace('.gz', '').replace('.raw', '').split('_')
    match len(rawfile_split):
        case 2:
            # this is an 'old old' file type
            out_dict['filename_spec'] = 1
            out_dict['dt'] = dt.strptime(filename.replace('.gz', '').replace('.raw', ''), '%Y%m%d%H%M%S_%f')
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
            gps_err = float(rawfile_split[6]) if ~np.isnan(out_dict['lon']) else 0
            if gps_err == -1:
                out_dict['embedded_gps_pps'] = True
            out_dict['gps_err'] = gps_err
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
        with open_gzip_or_dat(previous_file) as f:
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
    with open_gzip_or_dat(file_to_read) as f:
        ba = f.read()
    this_bytes = np.frombuffer(ba, dtype=np.uint8)
    return rotate_SA_array(this_bytes, packet_length=packet_length, previous_file=previous_file)


def decode_SA_array(data_array):
    # Extract the ADC local clock and convert the hexadecimal value to base 10
    # ADC clock is sent as 4 bytes, least significant byte first, so multiply by 256^0, 256^1, 256^2, 256^3
    adc_pps_micros = np.sum(data_array[:, 4:8] * (256 ** np.arange(4)), axis=1)
    # The feather's micros() function has a race condition where the us place can go from 999 to 0 before the ms place increments, so we need to correct for that
    race_condition_indices = np.nonzero(np.diff(adc_pps_micros) < -500)[0] + 1
    adc_pps_micros[race_condition_indices] += 1000
    # Extract the ADC reading and convert to decimal.
    # The ADC reading is sent as 3 bytes, most significant byte first, so multiply by 256^2, 256^1, 256^0
    adc_reading_dec = np.sum(data_array[:, 1:4] * np.flip(256 ** np.arange(3)), axis=1)
    # The ADC reading is sent as an unsigned 24-bit integer, so we need to convert it to a signed 23-bit integer
    # If the ADC reading is greater than 2^23 - 1, then it is a negative number
    adc_reading_overflow_mask = adc_reading_dec > (2**23 - 1)
    adc_reading = adc_reading_dec - adc_reading_overflow_mask.astype(int) * (2**24)
    return adc_pps_micros, adc_reading.astype(np.int32)

def decode_lsb_metadata(micros, adc_reading):
    # Decode metadata from hidden LSB data stream in ADC reading.
    lsb = adc_reading % 2
    lsb_changes = np.diff(lsb, prepend=lsb[0])
    lsb_change_indices = np.nonzero(lsb_changes)[0]
    lsb_rises = np.nonzero(lsb_changes == 1)[0]
    # The NMEA message is 30 bytes long, so if the signal goes high and doesn't change again for 30 bytes, it's a PPS signal
    pps_start_candidates = lsb_change_indices[(np.diff(lsb_change_indices, append=lsb_change_indices[-1]) > NMEA_MSG_SZ*8)]
    # Only look for rising edges more than 30 bytes after the last change
    pps_starts = np.intersect1d(pps_start_candidates, lsb_rises)
    # The end of the PPS is always the next change after the start (must be a falling edge)
    pps_ends = lsb_change_indices[np.searchsorted(lsb_change_indices, pps_starts, side='right')]
    # The start of the NMEA signal is always 11 bits after the end of the PPS
    nmea_starts = pps_ends + 11
    # The end of the NMEA signal is always 30 bytes after the start of the NMEA signal
    nmea_indices = np.arange(0, NMEA_MSG_SZ*8) + nmea_starts[:, np.newaxis]
    # The NMEA message is a constant data type, so unpack the structs
    nmea_dtypes = np.dtype([('epoch', '<u8'), ('micros_diff', 'u1'), ('lat', '<i4'), ('lon', '<i4'), ('alt', '<f4'), ('use_relay', 'u1'), ('cpu_id', '<u8')])
    nmea_data = np.packbits(lsb[nmea_indices].astype(np.uint8), axis=1, bitorder='big').view(nmea_dtypes)
    parsed_times = nmea_data['epoch'].astype('datetime64[s]').astype('datetime64[us]')
    # the PPS rising edge can only be relayed by the feather at the next ADC sample, but the delay time is recorded in the NMEA message
    parsed_times = parsed_times + nmea_data['micros_diff'].astype('timedelta64[us]') # this is the exact time of the PPS rising edge in the data we see
    parsed_times = np.squeeze(parsed_times[:, 0]) # flatten the array to 1D
    # we're about to interpolate times, but np.interp can't handle datetime64, so convert to integer microseconds since the first PPS
    us_offsets_from_first = (parsed_times - parsed_times[0]).astype('timedelta64[us]').astype(np.int64)
    # interpolate the PPS times to the ADC sample times
    pps_micros = np.interp(micros, micros[pps_starts], us_offsets_from_first)
    # the interpolation will only work between the first and last PPS, so extrapolate the start and end of the array
    time_between_samples_start = np.mean(np.diff(pps_micros[pps_starts[0]:pps_starts[0]+100])) # find the average time between samples at the start of the array
    pps_micros[0:pps_starts[0]] = np.arange(-pps_starts[0], 0) * time_between_samples_start + us_offsets_from_first[0] # linear extrapolation to the start of the array
    time_between_samples_end = np.mean(np.diff(pps_micros[pps_starts[-1]-100:pps_starts[-1]])) # find the average time between samples at the end of the array
    pps_micros[pps_starts[-1]:] = np.arange(0, pps_micros.size - pps_starts[-1]) * time_between_samples_end + us_offsets_from_first[-1] # linear extrapolation to the end of the array
    sample_times = pps_micros.astype('timedelta64[us]') + parsed_times[0] # convert back to datetime64
    sensor_lat = np.mean(np.squeeze(nmea_data['lat'])/1e7)
    sensor_lon = np.mean(np.squeeze(nmea_data['lon'])/1e7)
    sensor_alt = np.mean(np.squeeze(nmea_data['alt']))
    sensor_relay = nmea_data['use_relay'][0, 0]
    sensor_cpu = nmea_data['cpu_id'][0, 0]
    return sample_times, sensor_lat, sensor_lon, sensor_alt, sensor_relay, sensor_cpu
