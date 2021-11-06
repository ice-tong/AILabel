from commands import configs 
from multiprocessing import Pool
import subprocess


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source',
        '--srcdict', f'data-bin/ailabel_dicts_150k/{field}/dict.txt',
        '--trainpref', f'data-src/ailabel_150k/train.{field}',
        '--validpref', f'data-src/ailabel_150k/valid.{field}',
        '--destdir', f'data-bin/ailabel/{field}', '--workers', '40'
        ])


with Pool() as pool:
    pool.map(run, 
        configs.fields + [
            configs.static_field_label,
            configs.var_label_field
            ]
    )

