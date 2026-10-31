#!/usr/bin/env python3
# Common function "library" for slow antenna processing scripts
# Created 29 September 2025 by Sam Gardner <samuel.gardner@ttu.edu>

import numpy as np
from datetime import datetime as dt
import gzip

NMEA_MSG_SZ = 30
PPS_NMEA_DELAY = 11
NMEA_DTYPES = np.dtype([('epoch', '<u8'), ('micros_diff', 'u1'), ('lat', '<i4'), ('lon', '<i4'), ('alt', '<f4'), ('use_relay', 'u1'), ('cpu_id', '<u8')])

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
    """
    Parses metadata from slow antenna raw files' names.

    Parameters
    ----------
    filename : str
        The name of the file to parse.

    Returns
    -------
    out_dict : dict
        A dictionary containing the parsed metadata. Keys are:
            'filename_spec' : int
                The filename specification version. 1 = old old, 2 = old, 3 = current.
            'dt' : datetime
                The datetime of the file.
            'relay' : str
                The relay used for the data collection.
            'lon' : float
                The longitude of the sensor.
            'lat' : float
                The latitude of the sensor.
            'alt' : float
                The altitude of the sensor.
            'gps_err' : float
                The GPS error in meters.
            'cpu_id' : int
                The CPU ID of the sensor.
            'gzipped' : bool
                Whether the file is gzipped or not.
            'embedded_gps_pps' : bool
                Whether the file has embedded GPS PPS or not.
    """
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
            out_dict['alt'] = np.nan if rawfile_split[5].lower() == '2donly' else float(rawfile_split[5])
            gps_err = float(rawfile_split[6]) if ~np.isnan(out_dict['lon']) else 0
            if rawfile_split[6] == '-1.00':
                out_dict['embedded_gps_pps'] = True
            out_dict['gps_err'] = gps_err
            out_dict['cpu_id'] = int(rawfile_split[7], 16)
            out_dict['relay'] = rawfile_split[8]
        case _:
            raise ValueError(f'Filename {filename} does not match any known filename specifications.')
    return out_dict



def rotate_SA_array(this_bytes, packet_length=9, previous_file=None):
    """
    Rotates a 1D array of bytes into a 2D array of ADC packets, each of length `packet_length`. The first column is the header byte (0xBE), the last column is the footer byte (0xEF), and the middle columns contain the message data.
    
    
    Parameters
    ----------
    this_bytes : np.ndarray
        A 1D array of bytes to be rotated into a 2D array of messages.
    packet_length : int, optional
        The length of each message packet. Default is 9.
    previous_file : str, optional
        The path to the previous file in the sequence for recovering the first packet if it is split across files. Default is None.
    
    Returns
    -------
    adc_packets : np.ndarray
        Rotated 2D array of ADC packets, each of length `packet_length`.
    """
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
    """
    Reads a slow antenna raw file and returns a 2D array of ADC packets.

    Parameters
    ----------
    file_to_read : str
        The path to the file to read.
    packet_length : int, optional
        The length of each message packet. Default is 9.
    previous_file : str, optional
        The path to the previous file in the sequence for recovering the first packet if it is split across files. Default is None.
    
    Returns
    -------
    adc_packets : np.ndarray
        Rotated 2D array of ADC packets, each of length `packet_length`.
    """
    with open_gzip_or_dat(file_to_read) as f:
        ba = f.read()
    this_bytes = np.frombuffer(ba, dtype=np.uint8)
    return rotate_SA_array(this_bytes, packet_length=packet_length, previous_file=previous_file)


def correct_micros(adc_micros):
    """
    Corrects for integer rollovers and microsecond clock race conditions in Feather M4 microseconds clock data.

    Parameters
    ----------
    adc_micros : np.ndarray
        Array of microsecond clock to be corrected. The nominal sample time must be much less than 500 us.

    Returns
    -------
    micros_correct : np.ndarray
        Microseconds, corrected for rollovers and race conditions.
    """
    # The feather's micros() function has a race condition where the us place can go from 999 to 0 before the ms place increments, so we need to correct for that
    micros_correct = adc_micros.copy().astype(np.int64)
    micros_diff = np.diff(micros_correct)
    race_condition_indices = np.nonzero((micros_diff < -500) & (micros_diff > -1500))[0] + 1
    # this only catches the race condition if it happens on the second sample or later, so check the first sample
    if len(micros_correct) > 1:
        if micros_diff[0] > 990 and micros_diff[0] < 1500:
            race_condition_indices = np.insert(race_condition_indices, 0, 0)
    micros_correct[race_condition_indices] += 1000
    rollover_indices = np.nonzero(micros_diff < -2**31)[0] + 1
    rollover_correction = np.zeros_like(micros_correct)
    correction_factor = np.iinfo(np.uint32).max + 1
    rollover_correction[rollover_indices] = correction_factor
    micros_correct += np.cumsum(rollover_correction)
    return micros_correct


def decode_SA_array(data_array):
    """
    Decode a 2D array of microcontroller messages into a microsecond clock and ADC reading.
    
    Parameters
    ----------
    data_array : np.ndarray
        A 2D array of microcontroller messages, where each row is a message and each column is a byte in the message.
    
    Returns
    -------
    adc_micros : np.ndarray
        The ADC local clock in microseconds, corrected for rollovers and race conditions. See `correct_micros`.
    adc_reading : np.ndarray
        The ADC reading values, converted to signed 23-bit integers.
    """
    # Extract the ADC local clock and convert the hexadecimal value to base 10
    # ADC clock is sent as 4 bytes, least significant byte first, so multiply by 256^0, 256^1, 256^2, 256^3
    adc_micros = np.sum(data_array[:, 4:8] * (256 ** np.arange(4)), axis=1)
    # Extract the ADC reading and convert to decimal.
    # The ADC reading is sent as 3 bytes, most significant byte first, so multiply by 256^2, 256^1, 256^0
    adc_reading_dec = np.sum(data_array[:, 1:4] * np.flip(256 ** np.arange(3)), axis=1)
    # The ADC reading is sent as an unsigned 24-bit integer, so we need to convert it to a signed 23-bit integer
    # If the ADC reading is greater than 2^23 - 1, then it is a negative number
    adc_reading_overflow_mask = adc_reading_dec > (2**23 - 1)
    adc_reading = adc_reading_dec - adc_reading_overflow_mask.astype(int) * (2**24)
    return adc_micros, adc_reading.astype(np.int32)

def decode_lsb_metadata(micros, adc_reading):
    """
    Decode embedded metadata from the least significant bit of the ADC reading

    Parameters
    ----------
    micros : np.ndarray
        The ADC microsecond clock values, corrected for rollovers and race condition jumps. See `correct_micros`.
    adc_reading : np.ndarray
        The ADC reading values.
    
    Returns
    -------
    sample_times : np.ndarray
        The absolute times of the samples.
    sensor_lat : float
        The latitude of the sensor.
    sensor_lon : float
        The longitude of the sensor.
    sensor_alt : float
        The altitude of the sensor.
    sensor_relay : int
        The analog board relay energized at the time of data collection, 0 = a, 1 = b, 2 = c
    sensor_cpu : int
        The serial number of the raspberry pi used to collect the data.
    """

    def _array_mode(a):
        vals, counts = np.unique(a, return_counts=True)
        return vals[np.argmax(counts)]

    lsb = adc_reading % 2
    lsb_changes = np.diff(lsb, prepend=lsb[0])
    lsb_change_indices = np.nonzero(lsb_changes)[0]
    if lsb_change_indices.size == 0:
        raise ValueError('Cannot decode LSB metadata: No changes in LSB in data provided.')
    lsb_rises = np.nonzero(lsb_changes == 1)[0]
    # The NMEA message is 30 bytes long, so if the signal goes high and doesn't change again for 30 bytes, it's a PPS signal
    pps_start_candidates = lsb_change_indices[(np.diff(lsb_change_indices, append=lsb_change_indices[-1]) > NMEA_MSG_SZ*8)]
    # Only look for rising edges more than 30 bytes after the last change
    pps_starts = np.intersect1d(pps_start_candidates, lsb_rises)
    # The end of the PPS is always the next change after the start (must be a falling edge)
    pps_ends = lsb_change_indices[np.searchsorted(lsb_change_indices, pps_starts, side='right')]
    # The start of the NMEA signal is always 11 bits after the end of the PPS
    nmea_starts = pps_ends + PPS_NMEA_DELAY
    # The end of the NMEA signal is always 30 bytes after the start of the NMEA signal
    nmea_indices = np.arange(0, NMEA_MSG_SZ*8) + nmea_starts[:, np.newaxis]
    if nmea_indices.size == 0 or pps_starts.size < 2:
        raise ValueError('Cannot decode LSB metadata: No NMEA messages found in the data.')
    valid_decodes = np.all(nmea_indices < lsb.size, axis=1)
    micros_diff = np.diff(micros)
    nominal_dt = np.median(micros_diff)
    missed_packets_idx = np.nonzero(micros_diff > 1.5*nominal_dt)[0][np.newaxis, :]
    missed_pkt_in_window = (missed_packets_idx >= pps_starts[:, np.newaxis] - 1) & (missed_packets_idx <  nmea_indices[:, -1][:, np.newaxis])
    nmea_pps_missed_packets = missed_pkt_in_window.any(axis=1)
    valid_decodes = valid_decodes & ~nmea_pps_missed_packets
    # The NMEA message is a constant data type, so unpack the structs
    nmea_data = np.packbits(lsb[np.minimum(nmea_indices, (lsb.size - 1))].astype(np.uint8), axis=1, bitorder='big').view(NMEA_DTYPES)
    unparsed_times = nmea_data['epoch'].flatten()
    valid_decodes = valid_decodes & (unparsed_times > 1577836800) # must be after 2020-01-01
    pps_seconds = np.round((micros[pps_starts] - micros[pps_starts][0]) / np.median(np.diff(micros[pps_starts]))).astype(np.int64)
    time_diffs_agree = np.diff(unparsed_times.astype(np.int64)) == np.diff(pps_seconds)
    both_time_diffs_agree = np.zeros(pps_starts.size, dtype=bool)
    both_time_diffs_agree[1:] = time_diffs_agree
    both_time_diffs_agree[:-1] |= time_diffs_agree
    valid_decodes = valid_decodes & both_time_diffs_agree
    const_times = unparsed_times.astype(np.int64) - pps_seconds
    valid_decodes = valid_decodes & (const_times == _array_mode(const_times)) # the time between PPS should be constant
    # extract the rest of the data
    pps_delay = nmea_data['micros_diff'].flatten()
    valid_decodes = valid_decodes & (pps_delay <= nominal_dt + 5) # the PPS delay must be less than the time between samples
    sensor_lat_all = np.squeeze(nmea_data['lat'])/1e7
    sensor_lat = np.nanmedian(sensor_lat_all)
    valid_decodes = valid_decodes & (np.abs(sensor_lat_all - sensor_lat) < 0.1)
    sensor_lon_all = np.squeeze(nmea_data['lon'])/1e7
    sensor_lon = np.nanmedian(sensor_lon_all)
    valid_decodes = valid_decodes & (np.abs(sensor_lon_all - sensor_lon) < 0.1)
    sensor_alt_all = np.squeeze(nmea_data['alt'])
    sensor_alt = np.nanmedian(sensor_alt_all)
    valid_decodes = valid_decodes & (np.abs(sensor_alt_all - sensor_alt) < 10)
    sensor_relay_all = np.squeeze(nmea_data['use_relay'])
    sensor_relay = _array_mode(sensor_relay_all)
    valid_decodes = valid_decodes & (sensor_relay_all == sensor_relay)
    sensor_cpu_all = np.squeeze(nmea_data['cpu_id'])
    sensor_cpu = _array_mode(sensor_cpu_all)
    valid_decodes = valid_decodes & (sensor_cpu_all == sensor_cpu)
    parsed_times = unparsed_times.astype('datetime64[s]').astype('datetime64[us]')[valid_decodes]
    if np.sum(valid_decodes) < 2:
        raise ValueError('Cannot decode LSB metadata: Not enough valid NMEA packets for time interpolation.')
    # the PPS rising edge can only be relayed by the feather at the next ADC sample, but the delay time is recorded in the NMEA message
    parsed_times = parsed_times + pps_delay[valid_decodes].flatten().astype('timedelta64[us]') # this is the exact time of the PPS rising edge in the data we see
    # we're about to interpolate times, but np.interp can't handle datetime64, so convert to integer microseconds since the first PPS
    us_offsets_from_first = (parsed_times - parsed_times[0]).astype('timedelta64[us]').astype(np.int64)
    # interpolate the PPS times to the ADC sample times, filtered to only the valid decodes
    valid_pps_starts = pps_starts[valid_decodes]
    pps_micros = np.interp(micros, micros[valid_pps_starts], us_offsets_from_first)
    # the interpolation will only work between the first and last PPS, so extrapolate the start and end of the array
    before_pps_drift = (us_offsets_from_first[1] - us_offsets_from_first[0]) / (micros[valid_pps_starts[1]] - micros[valid_pps_starts[0]]) # average clock drift between the first two valid PPS pulses
    pps_micros[:valid_pps_starts[0]] = us_offsets_from_first[0] + before_pps_drift * (micros[:valid_pps_starts[0]] - micros[valid_pps_starts[0]])
    after_pps_drift = (us_offsets_from_first[-1] - us_offsets_from_first[-2]) / (micros[valid_pps_starts[-1]] - micros[valid_pps_starts[-2]]) # average clock drift between the last two valid PPS pulses
    pps_micros[valid_pps_starts[-1]:] = us_offsets_from_first[-1] + after_pps_drift * (micros[valid_pps_starts[-1]:] - micros[valid_pps_starts[-1]])
    sample_times = pps_micros.astype('timedelta64[us]') + parsed_times[0] # convert back to datetime64
    return sample_times, sensor_lat, sensor_lon, sensor_alt, sensor_relay, sensor_cpu
