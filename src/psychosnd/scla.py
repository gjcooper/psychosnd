# coding: utf-8
from __future__ import division
from copy import deepcopy
from prespy.sndan import extract_sound_events, ExtractError, timing, stdStats
from .psychcsv import load

datasets = {'Port_to_Snd': [], 'Port_to_Port': None, 'Snd_to_Snd': None, 'Port_Length': None}


def scla(soundfile=None, logfile=None, **kwargs):
    """Implements similar logic to Neurobehavioural Systems SCLA program"""
    log = load(logfile)

    fs, pcodes, snds, port = extract_sound_events(soundfile, **kwargs)
    if (len(log.events) != len(pcodes)) or (len(pcodes) != len(snds)):
        raise ExtractError(log.events, pcodes, snds)

    thisdata = deepcopy(datasets)

    for code, snd in zip(pcodes, snds):
        thisdata['Port_to_Snd'].append(snd - code)
    td, pl = timing(port, pcodes, snds, fs, **kwargs)
    thisdata['Port_to_Port'] = td['pcodes']
    thisdata['Snd_to_Snd'] = td['snds']
    thisdata['Port_Length'] = pl
#    import pdb; pdb.set_trace()
    return stdStats(thisdata)
