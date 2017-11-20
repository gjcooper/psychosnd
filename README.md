psychosnd
===============================

version number: 0.0.7
author: Gavin Cooper

Overview
--------

Psychopy sound analysis scripts.

It contains one module that is a thin wrapper on the csv module to open a [Psychopy](http://www.psychopy.org/) csv data file and provide an interface similar to the [prespy](https://github.com/gjcooper/prespy) library.

The other module relies heavily on the sound card latency analyser functionality of the same `prespy` library in order to provide a similar interface as a command line script.


Installation / Usage
--------------------

To install use pip:

    $ pip install psychosnd

If you have a local version you can also use:

    $ pip install <path_to_package>/psychosnd-0.0.7.tar.gz


To use the command line script, once installed you can run `psych-scla <soundfile> <logfile>` to analyse the difference in sound presentation times as recorded in each file.

For the full procedure see the write up in the `prespy` documentation [here](https://github.com/gjcooper/prespy#scla-information).

For more information on the options you can pass to the scla command you can type in `psych-scla --help` for the full list.

Contributing
------------

TBD

Example
-------

TBD
