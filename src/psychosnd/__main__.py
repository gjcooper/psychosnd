from psychosnd.scla import scla, ExtractError, datasets
from psychosnd.__about__ import __version__, __title__
import sys
import argparse


def scla_script():
    parser = argparse.ArgumentParser(
        description='scla {} ({} variant) - Analyse sound latencies'.format(__version__, __title__),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--runid', '-i', help='An id for the sound plot', default='scla')
    parser.add_argument('soundfile', help='A sound recording of port and sound output')
    parser.add_argument('logfile', help='Stimulus delivery recording of event sequence')
    parser.add_argument('--schannel', '-c', help='The channel sounds were recorded in', default=1, type=int)
    parser.add_argument('--portdur', '-d', help='Max duration for sound/port event', default=0.012, type=float)
    parser.add_argument('--snddur', '-m', help='Max duration for sound/port event', default=0.012, type=float)
    parser.add_argument('--portthresh', '-t', help='Threshold for sound/port detection', default=0.2, type=float)
    parser.add_argument('--sndthresh', '-s', help='Threshold for sound/port detection', default=0.2, type=float)
    parser.add_argument('--version', action='version', version='scla {} ({} variant)'.format(__version__, __title__))
    parser.add_argument('--precision', '-p', help='Limit the precision of reported values to this many decimal places', default=3, type=int)
    parser.add_argument('--results', '-r', help='Limit the results to only fields of interest', choices=datasets, default=list(datasets.keys()), nargs='+')

    cmdargs = vars(parser.parse_args())
    precision = cmdargs.pop('precision')
    reports = cmdargs.pop('results')

    try:
        res = scla(**cmdargs)
    except ExtractError as e:
        lengths = map(len, [e.logData, e.portData, e.sndData])
        print('{}\nLogEvts: {} PortEvts: {} SndEvts: {}'.format(e.message, *lengths))
        sys.exit(65)
    report = ['==================================']
    for result in res:
        if result not in reports:
            continue
        report.append(result)
        for measure in ['mean', 'min', 'max', 'stddev']:
            report.append('\t{0}: {1:.{2}f}'.format(measure, res[result][measure], precision))
        report.append('----------------------------------')
    report.append(report[0])

    print('\n'.join(report))
