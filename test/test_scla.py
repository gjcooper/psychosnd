"""Tests for psychosnd.scla — sound latency analysis."""

import pytest
from unittest.mock import MagicMock, patch

from prespy.sndan import ExtractError
from psychosnd.scla import scla, datasets


# A minimal 3-event PsychoPy log fixture
SAMPLE_LOG = """\
trial,stimulus
1,tone_A
2,tone_B
3,tone_A
"""


@pytest.fixture
def logfile(tmp_path):
    f = tmp_path / "log.csv"
    f.write_text(SAMPLE_LOG)
    return str(f)


def _mock_timing(pcodes, snds, pl):
    """Return a timing mock matching 3-event data."""
    td = {'pcodes': list(pcodes), 'snds': list(snds)}
    return td, pl


class TestSclaHappyPath:
    """scla() returns correct structure and values when event counts match."""

    @pytest.fixture(autouse=True)
    def patched(self, logfile):
        self.pcodes = [0.100, 0.200, 0.300]
        self.snds   = [0.105, 0.206, 0.304]
        self.td     = {'pcodes': [0.100, 0.100], 'snds': [0.101, 0.098]}
        self.pl     = [0.010, 0.011, 0.012]
        self.logfile = logfile

        with patch('psychosnd.scla.extract_sound_events',
                   return_value=(48000, self.pcodes, self.snds, MagicMock())) as mock_extract, \
             patch('psychosnd.scla.timing',
                   return_value=(self.td, self.pl)) as mock_timing:
            self.result = scla(soundfile='fake.wav', logfile=logfile)
            self.mock_extract = mock_extract
            self.mock_timing = mock_timing

    def test_returns_all_dataset_keys(self):
        assert set(self.result.keys()) == set(datasets.keys())

    def test_each_key_has_stats_fields(self):
        for key in datasets:
            for field in ('mean', 'min', 'max', 'stddev', 'rawdata'):
                assert field in self.result[key], f"missing {field!r} in {key!r}"

    def test_port_to_snd_raw_values(self):
        expected = [s - c for c, s in zip(self.pcodes, self.snds)]
        assert self.result['Port_to_Snd']['rawdata'] == expected

    def test_port_to_snd_mean(self):
        raw = self.result['Port_to_Snd']['rawdata']
        import statistics
        assert self.result['Port_to_Snd']['mean'] == pytest.approx(statistics.mean(raw))

    def test_port_to_port_raw_values(self):
        assert self.result['Port_to_Port']['rawdata'] == self.td['pcodes']

    def test_snd_to_snd_raw_values(self):
        assert self.result['Snd_to_Snd']['rawdata'] == self.td['snds']

    def test_port_length_raw_values(self):
        assert self.result['Port_Length']['rawdata'] == self.pl

    def test_extract_sound_events_called_with_soundfile(self):
        self.mock_extract.assert_called_once()
        args, kwargs = self.mock_extract.call_args
        assert args[0] == 'fake.wav'

    def test_timing_called_after_extract(self):
        self.mock_timing.assert_called_once()


class TestSclaExtractError:
    """scla() raises ExtractError when event-count mismatches occur."""

    def test_log_longer_than_pcodes(self, logfile):
        # log has 3 events, only 2 port codes detected
        with patch('psychosnd.scla.extract_sound_events',
                   return_value=(48000, [0.1, 0.2], [0.105, 0.205], MagicMock())):
            with pytest.raises(ExtractError):
                scla(soundfile='fake.wav', logfile=logfile)

    def test_log_shorter_than_pcodes(self, tmp_path):
        # log has 2 events, 3 port codes detected
        f = tmp_path / "short.csv"
        f.write_text("trial,stimulus\n1,A\n2,B\n")
        with patch('psychosnd.scla.extract_sound_events',
                   return_value=(48000, [0.1, 0.2, 0.3], [0.105, 0.205, 0.305], MagicMock())):
            with pytest.raises(ExtractError):
                scla(soundfile='fake.wav', logfile=str(f))

    def test_pcodes_snds_length_mismatch(self, logfile):
        # Matching log but pcodes/snds differ
        with patch('psychosnd.scla.extract_sound_events',
                   return_value=(48000, [0.1, 0.2, 0.3], [0.105, 0.205], MagicMock())):
            with pytest.raises(ExtractError):
                scla(soundfile='fake.wav', logfile=logfile)

    def test_extract_error_has_data_attributes(self, logfile):
        with patch('psychosnd.scla.extract_sound_events',
                   return_value=(48000, [0.1, 0.2], [0.105, 0.205], MagicMock())):
            with pytest.raises(ExtractError) as exc_info:
                scla(soundfile='fake.wav', logfile=logfile)
        err = exc_info.value
        assert hasattr(err, 'logData')
        assert hasattr(err, 'portData')
        assert hasattr(err, 'sndData')


class TestSclaDoesNotMutateGlobal:
    """scla() must not accumulate state into the module-level datasets dict."""

    def test_port_to_snd_remains_empty_list(self, logfile):
        from psychosnd.scla import datasets as ds
        original = list(ds['Port_to_Snd'])

        with patch('psychosnd.scla.extract_sound_events',
                   return_value=(48000, [0.1, 0.2, 0.3], [0.105, 0.205, 0.305], MagicMock())), \
             patch('psychosnd.scla.timing',
                   return_value=({'pcodes': [0.1, 0.1], 'snds': [0.1, 0.1]}, [0.01, 0.01, 0.01])):
            scla(soundfile='fake.wav', logfile=logfile)

        assert ds['Port_to_Snd'] == original

    def test_repeated_calls_do_not_grow_global(self, logfile):
        from psychosnd.scla import datasets as ds

        common = dict(
            soundfile='fake.wav',
            logfile=logfile,
        )
        patches = (
            patch('psychosnd.scla.extract_sound_events',
                  return_value=(48000, [0.1, 0.2, 0.3], [0.105, 0.205, 0.305], MagicMock())),
            patch('psychosnd.scla.timing',
                  return_value=({'pcodes': [0.1, 0.1], 'snds': [0.1, 0.1]}, [0.01, 0.01, 0.01])),
        )
        with patches[0], patches[1]:
            scla(**common)
            scla(**common)

        assert ds['Port_to_Snd'] == []
