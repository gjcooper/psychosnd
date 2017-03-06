import csv


class PsychoPyCSV():
    """A simple abstraction for an in memory PsychoPy csv output file"""
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._load(filename)

    def _load(self, filename):
        """load the file contents into the events attribute"""
        with open(filename) as fp:
            reader = csv.DictReader(fp)
            self.events = list(reader)


def load(csvfile):
    """Load a psychopy csv into memory and expose the rows as events"""
    return PsychoPyCSV(csvfile)
