# coding: utf-8
from __future__ import division
from prespy.scla.sndan import extract_sound_events, ExtractError, timing, stdStats
from .psychcsv import load


def scla(soundfile=None, logfile=None, **kwargs):
    """Implements similar logic to Neurobehavioural Systems SCLA program"""
    log = load(logfile)

    fs, pcodes, snds, port = extract_sound_events(soundfile, **kwargs)
    if (len(log.events) != len(pcodes)) or (len(pcodes) != len(snds)):
        raise ExtractError(log.events, pcodes, snds)

    datasets = {}
    datasets['Port Snd Diffs'] = []
    for code, snd in zip(pcodes, snds):
        datasets['Port Snd Diffs'].append(snd - code)
    td, pl = timing(port, pcodes, snds, fs, **kwargs)
    datasets['Port Time Diffs'] = td['pcodes']
    datasets['Snd Time Diffs'] = td['snds']
    datasets['Port Code Lengths'] = pl
#    import pdb; pdb.set_trace()
    return stdStats(datasets)
