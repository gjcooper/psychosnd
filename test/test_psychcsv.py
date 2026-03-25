"""Tests for psychosnd.psychcsv — PsychoPy CSV loading."""

import pytest

from psychosnd.psychcsv import PsychoPyCSV, load


SAMPLE_CSV = """\
trial,stimulus,response,rt
1,tone_A,left,0.456
2,tone_B,right,0.389
3,tone_A,left,0.512
"""

EMPTY_CSV = """\
trial,stimulus,response,rt
"""


@pytest.fixture
def csv_file(tmp_path):
    f = tmp_path / "test_log.csv"
    f.write_text(SAMPLE_CSV)
    return str(f)


@pytest.fixture
def empty_csv_file(tmp_path):
    f = tmp_path / "empty_log.csv"
    f.write_text(EMPTY_CSV)
    return str(f)


class TestPsychoPyCSV:
    def test_events_count(self, csv_file):
        p = PsychoPyCSV(csv_file)
        assert len(p.events) == 3

    def test_events_are_dicts(self, csv_file):
        p = PsychoPyCSV(csv_file)
        assert all(isinstance(e, dict) for e in p.events)

    def test_correct_column_names(self, csv_file):
        p = PsychoPyCSV(csv_file)
        assert set(p.events[0].keys()) == {'trial', 'stimulus', 'response', 'rt'}

    def test_first_row_values(self, csv_file):
        p = PsychoPyCSV(csv_file)
        assert p.events[0]['trial'] == '1'
        assert p.events[0]['stimulus'] == 'tone_A'
        assert p.events[0]['response'] == 'left'
        assert p.events[0]['rt'] == '0.456'

    def test_second_row_values(self, csv_file):
        p = PsychoPyCSV(csv_file)
        assert p.events[1]['stimulus'] == 'tone_B'
        assert p.events[1]['response'] == 'right'

    def test_filename_stored(self, csv_file):
        p = PsychoPyCSV(csv_file)
        assert p.filename == csv_file

    def test_empty_csv_has_no_events(self, empty_csv_file):
        p = PsychoPyCSV(empty_csv_file)
        assert p.events == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PsychoPyCSV(str(tmp_path / "nonexistent.csv"))

    def test_reload_reflects_updated_file(self, tmp_path):
        # Writing a new file and loading it produces the right number of events
        f = tmp_path / "two_row.csv"
        f.write_text("trial,stimulus\n1,A\n2,B\n")
        p = PsychoPyCSV(str(f))
        assert len(p.events) == 2


class TestLoad:
    def test_returns_psychopycsv_instance(self, csv_file):
        result = load(csv_file)
        assert isinstance(result, PsychoPyCSV)

    def test_events_populated(self, csv_file):
        result = load(csv_file)
        assert len(result.events) == 3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(str(tmp_path / "missing.csv"))
