#!/usr/bin/env python3
"""
Apply custom envelopes (from a black & white image) to a segment of an audio file.
Includes an optional vertical offset for the envelope image.

Usage example:
    python apply_image_envelope.py input.wav envelope.png 3.5 --duration 1.2 --output out.wav --img_offset -10
This takes "input.wav", starting at 3.5 seconds for 1.2 seconds, and forces that segment’s
waveform to follow the envelope shape drawn in "envelope.png" after shifting the image vertically.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert
from PIL import Image
import argparse
from scipy.ndimage import maximum_filter1d, minimum_filter1d

def extract_envelopes(image_path, fill_concavities=True, window_size=7, vertical_offset=0):
    """
    Loads a black & white image, thresholds it,
    applies a vertical offset (padding with white),
    and extracts two envelope curves:
      - Top envelope (upper half): Uses the top‑most black pixel in each column.
      - Bottom envelope (lower half): Uses the bottom‑most black pixel in each column.
      
    The values are normalized so that:
      - Top envelope: 1 corresponds to the very top (row 0) and 0 to the midline.
      - Bottom envelope: -1 corresponds to the very bottom and 0 to the midline.
      
    Optionally applies a sliding-window filter to fill in concavities.
    
    Args:
        image_path (str): Path to the input image.
        fill_concavities (bool): If True, apply maximum/minimum filtering.
        window_size (int): Sliding window size for concavity filling.
        vertical_offset (int): Number of pixels to shift the image vertically.
                              Positive values shift downward; negative upward.
                              Padding will be white.
    
    Returns:
        E_top (np.array): 1D array (length = image width) for the top envelope.
        E_bottom (np.array): 1D array for the bottom envelope.
        arr (np.array): The (potentially shifted and thresholded) image array.
    """
    # Load image in grayscale and threshold: pixels < 128 become black (0), others white (255)
    im = Image.open(image_path).convert('L')
    im = im.point(lambda p: 0 if p < 128 else 255)
    arr = np.array(im)
    
    # Apply vertical offset if needed, with white padding.
    if vertical_offset != 0:
        H, W = arr.shape
        new_arr = np.full((H, W), 255, dtype=arr.dtype)  # white = 255
        if vertical_offset > 0:
            # Shift downward: top rows become white.
            new_arr[vertical_offset:, :] = arr[:H-vertical_offset, :]
        else:
            # Shift upward: bottom rows become white.
            new_arr[:H+vertical_offset, :] = arr[-vertical_offset:, :]
        arr = new_arr

    H, W = arr.shape
    mid = H // 2
    E_top = np.zeros(W)
    E_bottom = np.zeros(W)
    
    for x in range(W):
        # --- Top envelope: use the top‑most black pixel in the upper half ---
        col_top = arr[:mid, x]
        black_idx_top = np.where(col_top == 0)[0]
        if black_idx_top.size > 0:
            y_top = black_idx_top[0]
        else:
            y_top = mid  # default to midline if no black pixel found
        # Normalize so that y_top==0 gives 1 and y_top==mid gives 0.
        E_top[x] = (mid - y_top) / mid

        # --- Bottom envelope: use the bottom‑most black pixel in the lower half ---
        col_bottom = arr[mid:, x]
        black_idx_bottom = np.where(col_bottom == 0)[0]
        if black_idx_bottom.size > 0:
            y_bottom = black_idx_bottom[-1] + mid  # adjust index offset
        else:
            y_bottom = mid
        # Normalize so that y_bottom==mid gives 0 and y_bottom==H-1 gives -1.
        E_bottom[x] = - (y_bottom - mid) / mid

    # Optionally fill in concavities with a sliding-window filter.
    if fill_concavities:
        E_top = maximum_filter1d(E_top, size=window_size, mode='reflect')
        E_bottom = minimum_filter1d(E_bottom, size=window_size, mode='reflect')
    
    return E_top, E_bottom, arr

def interpolate_envelope(envelope, target_length):
    """
    Linearly interpolate a 1D envelope array to match the target length.
    
    Args:
        envelope (np.array): Input envelope array.
        target_length (int): Desired output length.
    
    Returns:
        interp_env (np.array): Interpolated envelope.
    """
    x_old = np.linspace(0, 1, len(envelope))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, envelope)

def process_audio(audio_path, image_path, start_time, envelope_duration, output_path,
                  fill=True, window_size=7, img_offset=0):
    """
    Applies the envelopes from the (optionally offset) image to a segment of the audio file.
    
    Steps:
      1. Read the audio file (assumes 16-bit PCM; adjust as needed).
      2. Extract and interpolate the top and bottom envelopes from the image.
         (The image is shifted vertically by img_offset pixels before extraction.)
      3. Extract the audio segment (starting at start_time for envelope_duration seconds).
      4. Use a Hilbert transform to get the instantaneous phase (carrier) of the segment.
      5. Remap the carrier so that:
             new_sample = E_bottom_interp + ((carrier+1)/2) * (E_top_interp_offset - E_bottom_interp)
         A small DC offset is added to the top envelope to avoid cancellation.
      6. Replace the audio segment with the modified waveform and write out the result.
    """
    # Load audio
    sr, data = wavfile.read(audio_path)
    if data.ndim > 1:
        data = data[:, 0]  # take first channel if stereo

    # Normalize audio to float [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0

    start_sample = int(start_time * sr)
    N_env = int(envelope_duration * sr)

    # Extract envelopes from image with the optional vertical offset.
    E_top, E_bottom, _ = extract_envelopes(image_path, fill_concavities=fill,
                                           window_size=window_size, vertical_offset=img_offset)
    # Interpolate envelopes to match the number of samples in our audio segment.
    E_top_interp = interpolate_envelope(E_top, N_env)
    E_bottom_interp = interpolate_envelope(E_bottom, N_env)

    # Extract the target audio segment.
    segment = data[start_sample:start_sample + N_env]
    if len(segment) < N_env:
        segment = np.pad(segment, (0, N_env - len(segment)), mode='constant')

    # Use the Hilbert transform to extract the instantaneous phase (carrier).
    analytic_signal = hilbert(segment)
    phase = np.angle(analytic_signal)
    carrier = np.sin(phase)  # carrier in [-1, 1]

    # Add a small DC offset to the top envelope to avoid perfect cancellation.
    dc_offset = 0.01
    E_top_offset = E_top_interp + dc_offset

    # Map the carrier into the envelope range.
    new_segment = E_bottom_interp + ((carrier + 1) / 2) * (E_top_offset - E_bottom_interp)

    # Replace the original audio segment with the new one.
    new_data = np.copy(data)
    new_data[start_sample:start_sample + N_env] = new_segment

    # Denormalize and write the output (16-bit PCM).
    new_data_int16 = np.int16(np.clip(new_data, -1.0, 1.0) * 32767)
    wavfile.write(output_path, sr, new_data_int16)
    print("Output written to:", output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Apply an image-based envelope to a segment of an audio file with an optional vertical image offset."
    )
    parser.add_argument("audio_path", help="Path to the input WAV audio file")
    parser.add_argument("image_path", help="Path to the black & white envelope image")
    parser.add_argument("start_time", type=float,
                        help="Start time (in seconds) where the envelope is applied")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Duration (in seconds) of the envelope application (default: 1.0)")
    parser.add_argument("--output", default="output.wav", help="Output WAV file name (default: output.wav)")
    parser.add_argument("--fill", action="store_true", default=True,
                        help="Apply concavity filling (default: True)")
    parser.add_argument("--window", type=int, default=7,
                        help="Sliding window size for concavity filling (default: 7)")
    parser.add_argument("--img_offset", type=int, default=0,
                        help="Vertical offset (in pixels) for the envelope image. Positive shifts down, negative shifts up (default: 0)")
    args = parser.parse_args()
    
    process_audio(args.audio_path, args.image_path, args.start_time,
                  args.duration, args.output, fill=args.fill,
                  window_size=args.window, img_offset=args.img_offset)

if __name__ == '__main__':
    main()
