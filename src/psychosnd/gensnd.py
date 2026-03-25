"""Generate .wav tone files matching the PsySound MATLAB SoundSpec/Sound interface."""

import argparse
import numpy as np
from scipy.io import wavfile


def _hanning_half(n):
    """Return the ascending half of MATLAB's hanning(2n) window.

    MATLAB's symmetric hanning uses denominator (N+1), unlike numpy/scipy which
    use (N-1). This replicates calc_hanning(n, 2n) = 0.5*(1-cos(2pi*k/(2n+1)))
    for k=1..n, which forms the fade-in portion of the rcos ramp.
    """
    k = np.arange(1, n + 1)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * k / (2 * n + 1)))


def _rcos(sig, p1, p2):
    """Apply a raised cosine ramp to the start and/or end of sig.

    Matches PsySound.rcos(sig, P1, P2): p1 and p2 are the percentages of the
    signal length over which to apply the fade-in and fade-out ramps.
    """
    r = len(sig)
    out = sig.copy()
    if p1 != 0:
        fi = round(r * (p1 / 100))
        out[:fi] *= _hanning_half(fi)
    if p2 != 0:
        fo = round(r * (p2 / 100))
        out[r - fo:] *= _hanning_half(fo)[::-1]
    return out


def generate_tone(
    frequency=1000.0,
    duration=1.0,
    sample_rate=48000,
    amplitude=0.9,
    ramp_start=0.005,
    ramp_end=0.005,
    sound_type='norm',
    ear='na',
    delay=0.0,
):
    """Generate a stereo tone matching PsySound.Sound(spec).

    Parameters
    ----------
    frequency : float
        Tone frequency in Hz.
    duration : float
        Duration in seconds.
    sample_rate : int
        Sample rate in Hz.
    amplitude : float
        Peak amplitude, 0–1.
    ramp_start : float
        Fade-in ramp length in seconds.
    ramp_end : float
        Fade-out ramp length in seconds.
    sound_type : str
        One of 'norm', 'phase', 'time', 'sing'. Matches SoundSpec.Type.
    ear : str
        Target ear for asymmetric types: 'left', 'right', or 'na'.
    delay : float
        Interaural delay in seconds (used with 'phase' and 'time' types).

    Returns
    -------
    numpy.ndarray, shape (n_samples, 2)
        Stereo audio data in the range [-amplitude, amplitude].
    """
    bins = round(duration * sample_rate)
    dbins = round(delay * sample_rate)
    numcycles = frequency * duration

    rfisteps = (ramp_start / duration) * 100
    rfosteps = (ramp_end / duration) * 100

    t = np.linspace(0, numcycles * 2 * np.pi, bins)
    raw = np.sin(t)
    chan1 = amplitude * _rcos(raw, rfisteps, rfosteps)

    if sound_type == 'phase':
        start_phase = delay * frequency
        t2 = np.linspace(
            start_phase * 2 * np.pi,
            start_phase * 2 * np.pi + numcycles * 2 * np.pi,
            bins,
        )
        chan2 = amplitude * _rcos(np.sin(t2), rfisteps, rfosteps)
        data = np.column_stack([chan1, chan2] if ear == 'left' else [chan2, chan1])

    elif sound_type == 'sing':
        chan2 = np.zeros(bins)
        data = np.column_stack([chan1, chan2] if ear == 'left' else [chan2, chan1])

    elif sound_type == 'time':
        chan2 = amplitude * _rcos(raw, rfisteps, rfosteps)
        chan1_padded = np.concatenate([chan1, np.zeros(dbins)])
        chan2_padded = np.concatenate([chan2, np.zeros(dbins)])
        chan2d = np.concatenate([np.zeros(dbins), chan2_padded[: len(chan2_padded) - dbins]])
        data = np.column_stack(
            [chan2d, chan1_padded] if ear == 'left' else [chan1_padded, chan2d]
        )

    else:  # 'norm' (and fallback for 'complex')
        chan2 = amplitude * _rcos(raw, rfisteps, rfosteps)
        data = np.column_stack([chan1, chan2])

    return data


def save_wav(filename, data, sample_rate, bitrate=16):
    """Write stereo float data to a wav file.

    Parameters
    ----------
    filename : str
        Output file path.
    data : numpy.ndarray, shape (n_samples, 2)
        Audio data in range [-1, 1].
    sample_rate : int
        Sample rate in Hz.
    bitrate : int
        Bit depth: 16 (int16 PCM), 24 (int32 PCM), or 32 (float32).
    """
    if bitrate == 16:
        out = (data * 32767).astype(np.int16)
    elif bitrate == 24:
        out = (data * 2147483647).astype(np.int32)
    else:
        out = data.astype(np.float32)
    wavfile.write(filename, sample_rate, out)


def gensnd_script():
    parser = argparse.ArgumentParser(
        description='Generate .wav tone files for use in PsychoPy experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'durations',
        nargs='+',
        type=float,
        metavar='DURATION_MS',
        help='One or more tone durations in milliseconds',
    )
    parser.add_argument(
        '--frequency', '-f',
        type=float,
        default=1000.0,
        help='Tone frequency in Hz',
    )
    parser.add_argument(
        '--sample-rate', '-r',
        type=int,
        default=48000,
        help='Sample rate in Hz',
    )
    parser.add_argument(
        '--amplitude', '-a',
        type=float,
        default=0.9,
        help='Peak amplitude (0–1)',
    )
    parser.add_argument(
        '--ramp-start',
        type=float,
        default=5.0,
        help='Fade-in ramp length in ms',
    )
    parser.add_argument(
        '--ramp-end',
        type=float,
        default=5.0,
        help='Fade-out ramp length in ms',
    )
    parser.add_argument(
        '--bitrate', '-b',
        type=int,
        default=16,
        choices=[16, 24, 32],
        help='Bit depth of output wav',
    )
    parser.add_argument(
        '--type', '-t',
        dest='sound_type',
        choices=['norm', 'phase', 'time', 'sing'],
        default='norm',
        help='Sound generation type',
    )
    parser.add_argument(
        '--ear', '-e',
        choices=['left', 'right', 'na'],
        default='na',
        help='Target ear for asymmetric types (phase, time, sing)',
    )
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=0.0,
        help='Interaural delay in ms (used with --type time or phase)',
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help=(
            'Output filename pattern. Use {freq} and {dur} as placeholders. '
            'Defaults to "{freq}Hz{dur}ms.wav"'
        ),
    )

    args = parser.parse_args()
    output_pattern = args.output or '{freq}Hz{dur}ms.wav'

    freq = args.frequency
    freq_label = int(freq) if freq == int(freq) else freq

    for dur_ms in args.durations:
        dur_label = int(dur_ms) if dur_ms == int(dur_ms) else dur_ms
        data = generate_tone(
            frequency=freq,
            duration=dur_ms / 1000.0,
            sample_rate=args.sample_rate,
            amplitude=args.amplitude,
            ramp_start=args.ramp_start / 1000.0,
            ramp_end=args.ramp_end / 1000.0,
            sound_type=args.sound_type,
            ear=args.ear,
            delay=args.delay / 1000.0,
        )
        filename = output_pattern.format(freq=freq_label, dur=dur_label)
        save_wav(filename, data, args.sample_rate, args.bitrate)
        print(filename)


if __name__ == '__main__':
    gensnd_script()
