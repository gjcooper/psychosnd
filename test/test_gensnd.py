"""Tests for psychosnd.gensnd — tone generation and wav saving."""

import numpy as np
import pytest
from scipy.io import wavfile

from psychosnd.gensnd import _hanning_half, _rcos, generate_tone, save_wav


class TestHanningHalf:
    """_hanning_half returns the ascending half of MATLAB's symmetric hanning window."""

    def test_length(self):
        for n in [1, 5, 10, 100]:
            assert len(_hanning_half(n)) == n

    def test_ascending(self):
        w = _hanning_half(100)
        assert np.all(np.diff(w) >= 0)

    def test_starts_near_zero(self):
        # First sample of a long window should be very close to 0
        assert _hanning_half(100)[0] < 0.02

    def test_ends_near_one(self):
        # Last sample of a long window should be very close to 1
        assert _hanning_half(100)[-1] > 0.98

    def test_matches_formula(self):
        # Exact MATLAB formula: 0.5*(1 - cos(2*pi*k/(2n+1))), k=1..n
        n = 10
        k = np.arange(1, n + 1)
        expected = 0.5 * (1.0 - np.cos(2.0 * np.pi * k / (2 * n + 1)))
        np.testing.assert_allclose(_hanning_half(n), expected)

    def test_n1(self):
        w = _hanning_half(1)
        expected = 0.5 * (1.0 - np.cos(2.0 * np.pi / 3.0))
        np.testing.assert_allclose(w[0], expected)

    def test_values_between_zero_and_one(self):
        w = _hanning_half(50)
        assert np.all(w >= 0)
        assert np.all(w <= 1)


class TestRcos:
    """_rcos applies raised-cosine ramps to a signal."""

    @pytest.fixture(autouse=True)
    def sig(self):
        self.sig = np.ones(100)

    def test_no_ramp_unchanged(self):
        result = _rcos(self.sig, 0, 0)
        np.testing.assert_array_equal(result, self.sig)

    def test_start_ramp_first_sample_is_small(self):
        # MATLAB's hanning(N) uses denominator N+1, so the first sample is not
        # exactly zero: 0.5*(1-cos(2*pi/(2*fi+1))) ≈ 0.022 for fi=10.
        result = _rcos(self.sig, 10, 0)
        assert result[0] < 0.05

    def test_end_ramp_last_sample_is_small(self):
        result = _rcos(self.sig, 0, 10)
        assert result[-1] < 0.05

    def test_start_ramp_tail_unmodified(self):
        # Samples beyond the ramp should stay at 1.0
        result = _rcos(self.sig, 10, 0)  # 10% of 100 = 10 samples ramped
        np.testing.assert_array_equal(result[10:], np.ones(90))

    def test_end_ramp_head_unmodified(self):
        result = _rcos(self.sig, 0, 10)  # last 10 samples ramped
        np.testing.assert_array_equal(result[:90], np.ones(90))

    def test_both_ramps_length_preserved(self):
        result = _rcos(self.sig, 10, 10)
        assert len(result) == len(self.sig)

    def test_both_ramps_endpoints_near_zero(self):
        result = _rcos(self.sig, 10, 10)
        assert result[0] < 0.05
        assert result[-1] < 0.05

    def test_does_not_modify_input(self):
        original = self.sig.copy()
        _rcos(self.sig, 10, 10)
        np.testing.assert_array_equal(self.sig, original)

    def test_ramp_is_ascending_at_start(self):
        result = _rcos(self.sig, 20, 0)  # 20 samples of ramp
        assert np.all(np.diff(result[:20]) >= 0)

    def test_ramp_is_descending_at_end(self):
        result = _rcos(self.sig, 0, 20)
        assert np.all(np.diff(result[-20:]) <= 0)

    def test_zero_length_ramp_percent(self):
        # p=0 on either side should leave that side untouched
        result = _rcos(self.sig, 0, 0)
        assert result[0] == 1.0
        assert result[-1] == 1.0


class TestGenerateTone:
    """generate_tone produces correctly shaped and scaled stereo audio."""

    def test_shape_norm_30ms(self):
        data = generate_tone(frequency=1000, duration=0.030, sample_rate=48000)
        assert data.shape == (1440, 2)

    def test_shape_uses_round(self):
        # duration * sample_rate = 1440 exactly for 30ms @ 48kHz
        data = generate_tone(duration=0.030, sample_rate=48000)
        assert data.shape[0] == round(0.030 * 48000)

    @pytest.mark.parametrize("dur_ms", [30, 60, 100, 200])
    def test_shape_various_durations(self, dur_ms):
        data = generate_tone(duration=dur_ms / 1000.0, sample_rate=48000)
        assert data.shape == (round(dur_ms / 1000.0 * 48000), 2)

    def test_norm_channels_identical(self):
        data = generate_tone(frequency=1000, duration=0.030, sample_rate=48000, sound_type='norm')
        np.testing.assert_array_equal(data[:, 0], data[:, 1])

    @pytest.mark.parametrize("amp", [0.5, 0.9, 1.0])
    def test_amplitude_respected(self, amp):
        # With no ramps the peak should be very close to amplitude
        data = generate_tone(
            frequency=1000, duration=0.1, sample_rate=48000,
            amplitude=amp, ramp_start=0.0, ramp_end=0.0,
        )
        assert np.max(np.abs(data)) == pytest.approx(amp, rel=0.01)

    def test_ramp_start_silences_first_sample(self):
        data = generate_tone(frequency=1000, duration=0.030, sample_rate=48000, ramp_start=0.005)
        np.testing.assert_array_equal(data[0], [0.0, 0.0])

    def test_ramp_end_silences_last_sample(self):
        data = generate_tone(frequency=1000, duration=0.030, sample_rate=48000, ramp_end=0.005)
        np.testing.assert_allclose(data[-1], [0.0, 0.0], atol=1e-15)

    def test_sing_left_ear(self):
        data = generate_tone(
            frequency=1000, duration=0.1, sample_rate=48000,
            sound_type='sing', ear='left', ramp_start=0.0, ramp_end=0.0,
        )
        assert np.max(np.abs(data[:, 0])) > 0          # left channel has tone
        np.testing.assert_array_equal(data[:, 1], np.zeros(data.shape[0]))  # right is silent

    def test_sing_right_ear(self):
        data = generate_tone(
            frequency=1000, duration=0.1, sample_rate=48000,
            sound_type='sing', ear='right', ramp_start=0.0, ramp_end=0.0,
        )
        np.testing.assert_array_equal(data[:, 0], np.zeros(data.shape[0]))  # left is silent
        assert np.max(np.abs(data[:, 1])) > 0          # right channel has tone

    def test_phase_channels_differ(self):
        data = generate_tone(
            frequency=1000, duration=0.1, sample_rate=48000,
            sound_type='phase', delay=0.0005,
        )
        assert not np.allclose(data[:, 0], data[:, 1])

    def test_time_shape_includes_delay(self):
        bins = round(0.030 * 48000)
        dbins = round(0.005 * 48000)
        data = generate_tone(
            frequency=1000, duration=0.030, sample_rate=48000,
            sound_type='time', delay=0.005,
        )
        assert data.shape == (bins + dbins, 2)

    def test_time_left_ear_delayed_channel_starts_silent(self):
        # ear='left': left channel = chan2d (delayed), first dbins samples must be zero
        dbins = round(0.005 * 48000)
        data = generate_tone(
            frequency=1000, duration=0.030, sample_rate=48000,
            sound_type='time', ear='left', delay=0.005,
            ramp_start=0.0, ramp_end=0.0,
        )
        np.testing.assert_array_equal(data[:dbins, 0], np.zeros(dbins))

    def test_time_right_ear_delayed_channel_starts_silent(self):
        # ear='right': right channel = chan2d (delayed)
        dbins = round(0.005 * 48000)
        data = generate_tone(
            frequency=1000, duration=0.030, sample_rate=48000,
            sound_type='time', ear='right', delay=0.005,
            ramp_start=0.0, ramp_end=0.0,
        )
        np.testing.assert_array_equal(data[:dbins, 1], np.zeros(dbins))

    def test_sample_rate_changes_shape(self):
        data_44 = generate_tone(duration=0.1, sample_rate=44100)
        data_48 = generate_tone(duration=0.1, sample_rate=48000)
        assert data_44.shape[0] == round(0.1 * 44100)
        assert data_48.shape[0] == round(0.1 * 48000)

    def test_output_is_float(self):
        data = generate_tone()
        assert data.dtype.kind == 'f'


class TestSaveWav:
    """save_wav writes correct wav files for different bit depths."""

    def test_creates_file(self, tmp_path):
        data = generate_tone(frequency=1000, duration=0.030, sample_rate=48000)
        out = tmp_path / "test.wav"
        save_wav(str(out), data, 48000)
        assert out.exists()

    def test_16bit_dtype(self, tmp_path):
        data = generate_tone(frequency=1000, duration=0.030, sample_rate=48000)
        out = tmp_path / "test.wav"
        save_wav(str(out), data, 48000, bitrate=16)
        rate, loaded = wavfile.read(str(out))
        assert loaded.dtype == np.int16

    def test_32bit_dtype(self, tmp_path):
        data = generate_tone(frequency=1000, duration=0.030, sample_rate=48000)
        out = tmp_path / "test.wav"
        save_wav(str(out), data, 48000, bitrate=32)
        rate, loaded = wavfile.read(str(out))
        assert loaded.dtype == np.float32

    def test_correct_sample_rate(self, tmp_path):
        data = generate_tone(frequency=1000, duration=0.030, sample_rate=48000)
        out = tmp_path / "test.wav"
        save_wav(str(out), data, 48000)
        rate, _ = wavfile.read(str(out))
        assert rate == 48000

    def test_correct_shape_roundtrip(self, tmp_path):
        data = generate_tone(frequency=1000, duration=0.030, sample_rate=48000)
        out = tmp_path / "test.wav"
        save_wav(str(out), data, 48000)
        _, loaded = wavfile.read(str(out))
        assert loaded.shape == (1440, 2)

    def test_16bit_amplitude_scaling(self, tmp_path):
        data = generate_tone(
            frequency=1000, duration=0.1, sample_rate=48000,
            amplitude=0.9, ramp_start=0.0, ramp_end=0.0,
        )
        out = tmp_path / "test.wav"
        save_wav(str(out), data, 48000, bitrate=16)
        _, loaded = wavfile.read(str(out))
        assert np.max(np.abs(loaded)) == pytest.approx(0.9 * 32767, rel=0.01)

    def test_different_sample_rates(self, tmp_path):
        for sr in [44100, 48000]:
            data = generate_tone(duration=0.05, sample_rate=sr)
            out = tmp_path / f"test_{sr}.wav"
            save_wav(str(out), data, sr)
            rate, _ = wavfile.read(str(out))
            assert rate == sr
