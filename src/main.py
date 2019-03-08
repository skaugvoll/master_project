import os, sys


import run_inference



_CSV_FORMAT = 'csv'
_CWA_FORMAT = 'cwa_7z'
_FORMATS = [_CSV_FORMAT, _CWA_FORMAT]

job = {
    'config':'../params/config.yml',
    'source':'../data/input',
    'name':'testSubject08.7z',
    'source_fmt': _CWA_FORMAT,
    'output':'../data/output',
    'a':''
}

working_dir = os.getcwd()

# Create areguments for run_inference script
args = run_inference.parser.parse_args([
    '-c', job['config'],
    '-f', os.path.join( job['source'], job['name'] ),
    '-F', job['source_fmt'],
    '-o', job['output'],
    '-w', working_dir,
    '--chunk-size', '20000',
    '--plot-daily-overview',
    '--plot-uncertainty-thresh', '0.4'
])

print(args, type(args))

# Run script
# run_inference.main( args )
