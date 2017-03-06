from psychosnd.scla import scla, ExtractError
from psychosnd.__about__ import __version__, __title__
import sys
import argparse


def scla_script():
    parser = argparse.ArgumentParser(description='scla {} ({} variant) - Analyse sound latencies'.format(__version__, __title__))
    parser.add_argument('--runid', '-i', help='An id for the sound plot', default='scla')
    parser.add_argument('soundfile', help='A sound recording of port and sound output')
    parser.add_argument('logfile', help='Stimulus delivery recording of event sequence')
    parser.add_argument('--schannel', '-c', help='The channel sounds were recorded in', default=1, type=int)
    parser.add_argument('--maxdur', '-m', help='Max duration for sound/port event', default=0.012, type=float)
    parser.add_argument('--thresh', '-t', help='Threshold for sound/port detection', default=0.2, type=float)
    parser.add_argument('--version', action='version', version='scla {} ({} variant)'.format(__version__, __title__))

    args = parser.parse_args()

    try:
        res = scla(**vars(args))
    except ExtractError as e:
        lengths = map(len, [e.logData, e.portData, e.sndData])
        print('{}\nLogEvts: {} PortEvts: {} SndEvts: {}'.format(e.message, *lengths))
        sys.exit(65)
    report = ['==================================']
    for result in res:
        report.append(result)
        for measure in ['mean', 'min', 'max', 'stddev']:
            report.append('\t{}: {}'.format(measure, res[result][measure]))
        report.append('----------------------------------')
    report.append(report[0])

    print('\n'.join(report))
