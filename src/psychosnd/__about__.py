from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("psychosnd")
except PackageNotFoundError:
    __version__ = "unknown"

__title__ = "psychosnd"
__summary__ = "Psychopy sound analysis scripts"
__uri__ = "https://github.com/gjcooper/psychosnd"
__license__ = "GPLv3"
__author__ = "Gavin Cooper"
__email__ = "gjcooper@gmail.com"
