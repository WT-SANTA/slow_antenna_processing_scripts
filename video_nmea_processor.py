import argparse
from datetime import timezone

import cv2
import ffmpeg
import numpy as np
import polars as pl
import pynmea2
import scipy.signal
from scipy.interpolate import interp1d


if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser('Parse GPS NMEA and PPS from a video\'s audio track')
    parser.add_argument('-i', '--input', required=True, help='Path to the input video file')
    parser.add_argument('-c', '--output-csv', default=None, help='Path to output a CSV file of frame numbers and times')
    parser.add_argument('-s', '--sample-rate', default=44100, type=int, help='Audio sample rate') # 44.1 kHz should be default for most
    parser.add_argument('-r', '--serial-bitrate', default=4800, type=int, help='Bitrate of the NMEA serial stream')
    parser.add_argument('-w', '--serial-bits-per-word', default=8, type=int, help='Bits per word of the NMEA serial stream')
    parser.add_argument('--serial-stopbits', default=1, type=int, help='Stop bits of the NMEA serial stream')
    parser.add_argument('-o', '--output-video', help='Path to the output video file')
    parser.add_argument('-t', default=None, type=str, help='Static text to overlay on all frames')
    args = parser.parse_args()
    input_video = args.input
    sample_rate = args.sample_rate
    num_channels = 2 # this was originally an arg but the rest of this script is really not designed to handle mono audio
    serial_bitrate = args.serial_bitrate
    serial_bits_per_word = args.serial_bits_per_word
    serial_stopbits = args.serial_stopbits

    # Run ffmpeg and capture raw PCM audio output
    print('Reading audio...')
    process = (
        ffmpeg.input(input_video)
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=num_channels, ar=sample_rate)
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Convert raw bytes to NumPy array
    audio_bytes = process[0]
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

    # Reshape to (num_samples, num_channels)
    audio_array = audio_array.reshape(-1, num_channels)

    # The PPS signal will have many more samples near zero than the GPS NMEA signal. Use this to determine which channel (L/R) is which.
    points_near_zero = np.sum(np.abs(audio_array) < 1000, axis=0)

    if np.abs(points_near_zero[0] - points_near_zero[1]) < audio_array.shape[0] * 0.5:
        raise ValueError('Unable to determine which channel is PPS and which is GPS NMEA.')
    else:
        if points_near_zero[0] > points_near_zero[1]:
            print('PPS appears to be left channel, NMEA appears to be right channel')
            pps_signal = audio_array[:, 0]
            gps_signal = audio_array[:, 1]
        else:
            print('PPS appears to be right channel, NMEA appears to be left channel')
            pps_signal = audio_array[:, 1]
            gps_signal = audio_array[:, 0]

    # Arrays of sample numbers and elapsed time since start of file
    sample_num = np.arange(audio_array.shape[0])
    sample_elapsed = sample_num / sample_rate

    # digitize the PPS signal
    pps_digitizer = np.max(np.abs(pps_signal)) / 4 # if the signal exceeds 25% of the max, it's probably a pulse
    pps_high_mask = np.abs(pps_signal) > pps_digitizer # All points where the PPS signal is in the mark state
    # Only consider "rising edge"... points that are at least 0.5 seconds since the last mark. The first mark is always considered a start.
    pps_digital_starts = np.append([True], (sample_num[pps_high_mask][1:] >= sample_num[pps_high_mask][:-1] + sample_rate * 0.5))
    # Make a mask of only the first points of the marks
    pps_digital_mask = np.zeros_like(pps_signal, dtype=bool)
    pps_digital_mask[pps_high_mask] = pps_digital_starts
    # Obtain the elapsed time and sample number of the PPS rising edges.
    pps_frames = sample_num[pps_digital_mask]
    pps_digital_elapsed = sample_elapsed[pps_digital_mask]
    pps_digital = pps_signal[pps_digital_mask]


    # digitize the NMEA stream
    gps_nonzero_mask = ~(np.abs(gps_signal) < np.max(np.abs(gps_signal)) / 10) # Mask of points where the GPS signal is not zero
    gps_nonzero_signal = gps_signal[gps_nonzero_mask]
    gps_nonzero_elapsed = sample_elapsed[gps_nonzero_mask]
    # Find the start of each NMEA packet by finding the first point where the signal is not close to zero
    gps_packet_starts = np.append([True], ((gps_nonzero_elapsed[1:] - gps_nonzero_elapsed[:-1]) > 0.1))
    gps_packet_ends = np.append(gps_packet_starts[1:], [False]) # The end of a packet is 1 index behind the start of the next packet
    first_gps_packet_start = np.nonzero(gps_packet_starts)[0][0]
    last_gps_packet_start = np.nonzero(gps_packet_starts)[0][-1]
    # Exclude the first and last packet start as they may be cut off by the start or end of the recording
    gps_packet_starts[first_gps_packet_start] = False
    gps_packet_starts[last_gps_packet_start] = False
    # Filter the serial signal to reduce noise
    filtwin = scipy.signal.windows.blackmanharris(sample_rate//100) # .01 second moving average, blackman-harris convolution kernel
    # This acts as a rolling average for comparing the serial signal to see if it is in mark or space
    # Due to the nature of audio sampling, the serial signal has some low frequency components, especially at the start
    # By comparing to a moving average instead of a fixed threshold or zero-crossing, we can avoid misclassifying bits
    gps_digitizer = scipy.signal.oaconvolve(gps_signal, filtwin/np.sum(filtwin), mode='same')
    gps_nonzero_digitizer = gps_digitizer[gps_nonzero_mask]
    gps_digital_mask = gps_nonzero_signal > gps_nonzero_digitizer # Mask of points where the GPS signal is in the mark state

    # A few useful constants used in the digitization process
    sec_per_bit = serial_bitrate**(-1)
    samples_per_bit = int(sample_rate * sec_per_bit)
    bits_per_word = 1 + serial_bits_per_word + serial_stopbits # 1 start bit + 8 data bits + 1 stop bit


    print('Digitizing NMEA stream...')
    nmea_infos = [] # List containing NMEA strings
    nmea_frames = [] # List containing the start frame of each NMEA string
    # With standard NMEA, this is 1 start bit, 8 data bits, and 1 stop bit, for a total of 10 bits
    gps_packet_ends = np.nonzero(gps_packet_ends)[0] # the endpoints for each packet
    bit_offsets = np.arange(0, bits_per_word * sec_per_bit, sec_per_bit)
    for i, gps_packet_start in enumerate(np.nonzero(gps_packet_starts)[0]):
        print(f'Digitizing packet {i+1}/{np.sum(gps_packet_starts)}...')
        gps_packet_end = gps_packet_ends[i+1] # the matching endpoint for this packet
        gps_packet_end_elapsed = gps_nonzero_elapsed[gps_packet_end] # the elapsed time of the end of this packet
        this_bit_elapsed = gps_nonzero_elapsed[gps_packet_start] + 0.5 * sec_per_bit # Set up for the while loop. Start at the first bit of the packet
        serial_string = '' # String to hold the NMEA messages once parsed
        while this_bit_elapsed < gps_packet_end_elapsed:
            # This loop iterates once per word (start bit + data bits + stop bit) of the NMEA packet
            bit_elapseds = this_bit_elapsed + bit_offsets
            # Find the closest sample to each bit elapsed time
            closest_bit = np.searchsorted(gps_nonzero_elapsed, bit_elapseds)
            byte = gps_digital_mask[closest_bit] # Define an array containing the word we want to decode
            # Ensure that the start bit is present and valid
            if byte[0] != 0:
                # If the start bit is not detected, increment the elapsed time by 1 bit time and try again. This seemed to work well in testing.
                this_bit_elapsed += sec_per_bit
                continue
            if byte[-1] != 1:
                # If the stop bit is not detected, check to see if we are at the end of the packet.
                # At this point, just try decoding what we have, and if it doesn't work, give up.
                break
            else:
                # If a start and stop bit are found, great! We can proceed with decoding.
                # Janky implementation of a PLL
                # Select 1 second around the stop bit
                stop_bit_elapseds = np.array([bit_elapseds[-1]-sec_per_bit, bit_elapseds[-1]+sec_per_bit])
                stop_bit_limits = np.searchsorted(gps_nonzero_elapsed, stop_bit_elapseds)
                if stop_bit_limits[0] == stop_bit_limits[1]:
                    # If the stop bit limits are the same, we are at the end of the packet and cannot adjust for clock drift. Just increment by 1 bit time.
                    this_bit_elapsed += sec_per_bit
                    continue
                # Find the points in the stop bit that are most clearly "signal high"
                unambig = gps_nonzero_signal[stop_bit_limits[0]:stop_bit_limits[1]] - gps_nonzero_digitizer[stop_bit_limits[0]:stop_bit_limits[1]] > 10000
                # Use the average of these points to adjust the bit time for clock drift
                this_bit_elapsed = np.mean(gps_nonzero_elapsed[stop_bit_limits[0]:stop_bit_limits[1]][unambig]) + sec_per_bit # This is the time of the start bit of the next word.
            ascii_char = chr(np.sum(byte[1:-1] * 2**np.arange(8)))
            serial_string += ascii_char # Add the decoded character to the string
        # Split the string into lines and add to the list of NMEA sentences from this packet
        new_infos = [this_string.replace('\r', '') for this_string in serial_string.split('\n') if this_string != '']
        # Add this packet's sentences to the list of NMEA sentences
        nmea_infos.extend(new_infos)
        # Add this packet's start frame to the list of NMEA sentence times
        nmea_frames.extend([gps_packet_start] * len(new_infos))


    nmea_times = []
    nmea_lats = []
    nmea_lons = []
    for info in nmea_infos:
        try:
            this_nmea_msg = pynmea2.parse(info)
            try:
                this_nmea_time = this_nmea_msg.datetime
            except Exception:
                this_nmea_time = None
            try:
                this_nmea_lat = this_nmea_msg.latitude
            except Exception:
                this_nmea_lat = None
            try:
                this_nmea_lon = this_nmea_msg.longitude
            except Exception:
                this_nmea_lon = None
        except (pynmea2.ChecksumError, pynmea2.ParseError) as e:
            this_nmea_msg = None
            this_nmea_time = None
            this_nmea_lat = None
            this_nmea_lon = None
        nmea_times.append(this_nmea_time)
        nmea_lats.append(this_nmea_lat)
        nmea_lons.append(this_nmea_lon)
        

    # Keep only the NMEA sentences with valid times, and drop duplicates
    unique_nmea_times = []
    seen_nmea_times = set()
    unique_nmea_frames = []
    unique_nmea_lats = []
    unique_nmea_lons = []
    total_success = 0
    for i, this_nmea_time in enumerate(nmea_times):
        if this_nmea_time is not None:
            this_nmea_frame = nmea_frames[i]
            this_nmea_lat = nmea_lats[i]
            this_nmea_lon = nmea_lons[i]
            this_nmea_time = this_nmea_time.astimezone(timezone.utc).replace(tzinfo=None)
            if this_nmea_time not in seen_nmea_times:
                seen_nmea_times.add(this_nmea_time)
                unique_nmea_times.append(this_nmea_time)
                unique_nmea_frames.append(this_nmea_frame)
                unique_nmea_lats.append(this_nmea_lat)
                unique_nmea_lons.append(this_nmea_lon)
                total_success += 1
    print(f'Parsed {total_success} unique NMEA sentences with valid times')
    unique_nmea_times = np.array(list(unique_nmea_times)).astype('datetime64[s]')
    unique_nmea_frames = np.array(unique_nmea_frames)
    unique_nmea_lats = np.array(unique_nmea_lats)
    unique_nmea_lons = np.array(unique_nmea_lons)
    # Find the elapsed time of the unique NMEA frames
    unique_nmea_elapsed = gps_nonzero_elapsed[unique_nmea_frames]


    print('Associating NMEA and PPS...')
    # Associate each reported NMEA time with the previous PPS pulse time
    previous_pps = np.searchsorted(pps_digital_elapsed, unique_nmea_elapsed) - 1
    actual_nmea_elapsed = pps_digital_elapsed[previous_pps]

    # Use ffmpeg to read the frame times from the video
    frame_elapsed_json = ffmpeg.probe(input_video, select_streams='v', show_entries='packet=pts_time')
    frame_elapsed = np.sort(np.array([packet['pts_time'] for packet in frame_elapsed_json['packets']], dtype=float))

    # Interpolate the NMEA times to the frame times
    tinterper = interp1d(actual_nmea_elapsed, unique_nmea_times.astype('datetime64[ns]').astype(float), fill_value='extrapolate')
    frame_absolute_times = tinterper(frame_elapsed).astype('datetime64[ns]')
    latinterper = interp1d(actual_nmea_elapsed, unique_nmea_lats, fill_value='extrapolate')
    frame_latitudes = latinterper(frame_elapsed)
    loninterper = interp1d(actual_nmea_elapsed, unique_nmea_lons, fill_value='extrapolate')
    frame_longitudes = loninterper(frame_elapsed)


    # Write the CSV if requested
    if args.output_csv is not None:
        print('Writing CSV...')
        pl.DataFrame({'frame' : np.arange(len(frame_absolute_times)), 'elapsed' : frame_elapsed, 'time' : frame_absolute_times,
                      'latitude' : frame_latitudes, 'longitude' : frame_longitudes}).write_csv(args.output_csv)
    
    # Overlay the times on the video if requested
    if args.output_video is not None:
        frame_texts = frame_absolute_times.astype(str)
        # Read the input video
        # The probe above already carries the stream info, so nothing needs to be decoded here.
        video_stream = next(s for s in frame_elapsed_json['streams'] if s['codec_type'] == 'video')
        frame_width, frame_height = int(video_stream['width']), int(video_stream['height'])
        fps_num, fps_den = video_stream['avg_frame_rate'].split('/')
        fps = int(fps_num) / int(fps_den)

        font, fontScale, lineThickness, lineType = cv2.FONT_HERSHEY_SIMPLEX, 1, 1, cv2.LINE_AA
        fontColor = (0, 0, 204, 255) # BGRA - the alpha channel is what lets ffmpeg composite this

        # Size the band from the widest line we will draw, so we pipe as few bytes as possible
        texts = [f'{frame_texts[i]} {frame_latitudes[i]:.5f} {frame_longitudes[i]:.5f}'
                 for i in range(len(frame_texts))]
        (text_width, text_height), baseline = cv2.getTextSize(max(texts, key=len), font, fontScale, lineThickness)
        if args.t is not None:
            text_width = max(text_width, cv2.getTextSize(args.t, font, fontScale, lineThickness)[0][0])
        line_height = text_height + baseline + 10
        band_width = text_width + 20
        band_height = line_height * (2 if args.t is not None else 1) + 10

        # ffmpeg decodes the video, composites our band over it, and encodes - all internally
        # threaded. Only the band crosses into Python, so no full frame is ever copied here.
        video_in = ffmpeg.input(input_video)
        band_in = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgra',
                               s=f'{band_width}x{band_height}', r=fps)
        composited = ffmpeg.overlay(video_in['v'], band_in, x=10, y=frame_height - band_height - 10,
                                    shortest=1, format='auto')
        ffmpeg_proc = (
            ffmpeg
            .output(composited, args.output_video, vcodec='libx264', preset='veryfast', crf=20,
                    pix_fmt='yuv420p', an=None) # an=None emits a bare -an, dropping the audio track
            .global_args('-loglevel', 'error')
            .overwrite_output()
            # stderr is deliberately left attached to the terminal: piping it while we write to
            # stdin risks filling ffmpeg's stderr buffer and deadlocking both processes.
            .run_async(pipe_stdin=True)
        )

        band = np.zeros((band_height, band_width, 4), dtype=np.uint8)
        for frame_index, text in enumerate(texts):
            print(f'Overlaying: {frame_index/len(texts)*100:.2f}%')
            band[:] = 0 # fully transparent background
            cv2.putText(band, text, (10, band_height - 10), font, fontScale, fontColor, lineThickness, lineType)
            if args.t is not None:
                cv2.putText(band, args.t, (10, band_height - 10 - line_height), font, fontScale, fontColor, lineThickness, lineType)
            ffmpeg_proc.stdin.write(band.tobytes())
        ffmpeg_proc.stdin.close()
        if ffmpeg_proc.wait() != 0:
            raise RuntimeError('ffmpeg failed while writing the overlay video')
