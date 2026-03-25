psychosnd
===============================

version number: 0.0.12
author: Gavin Cooper

Overview
--------

Psychopy sound analysis and generation scripts.

It contains a thin wrapper on the csv module to open a [Psychopy](http://www.psychopy.org/) csv data file and provide an interface similar to the [prespy](https://github.com/gjcooper/prespy) library.

Two command-line tools are provided:

- **`psych-scla`** — analyses the latency between port triggers and recorded sounds, relying on the sound card latency analyser in the [prespy](https://github.com/gjcooper/prespy) library.
- **`psych-gensnd`** — generates stereo `.wav` tone files suitable for use in PsychoPy experiments, with configurable frequency, duration, amplitude, ramp times, and channel routing.


Installation / Usage
--------------------

To install use pip:

    $ pip install psychosnd

Or with [uv](https://docs.astral.sh/uv/):

    $ uv tool install psychosnd


### psych-scla

Analyse the difference in sound presentation times between a sound recording and a PsychoPy log file:

    $ psych-scla <soundfile> <logfile>

For the full procedure see the write up in the `prespy` documentation [here](https://github.com/gjcooper/prespy#scla-information).

For all available options:

    $ psych-scla --help


### psych-gensnd

Generate `.wav` tone files from the command line. Durations are given in milliseconds:

    $ psych-gensnd 30 60 --frequency 1000

This produces `1000Hz30ms.wav` and `1000Hz60ms.wav` — stereo 16-bit PCM files at 48 kHz with 5 ms raised-cosine ramps applied by default.

The output filename pattern, sample rate, amplitude, bit depth, and sound type (normal, phase-shifted, interaural time delay, single-ear) are all configurable:

    $ psych-gensnd 50 100 -f 440 -r 44100 --type sing --ear left
    $ psych-gensnd 30 --output "{freq}Hz_{dur}ms_ild.wav"

For all available options:

    $ psych-gensnd --help


Contributing
------------

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup instructions and the release process.


Example
-------

Generate a set of tones at 1000 Hz for two durations, then analyse a recording:

    $ psych-gensnd 30 60 --frequency 1000
    $ psych-scla recording.wav experiment_log.csv
