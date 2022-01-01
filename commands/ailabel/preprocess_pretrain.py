from commands import configs 
from multiprocessing import Pool
import subprocess


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source',
        '--srcdict', f'data-bin/ailabel_dicts_300k/{field}/dict.txt',
        '--trainpref', f'data-src/ailabel_300k/train.{field}',
        '--validpref', f'data-src/ailabel_300k/valid.{field}',
        '--destdir', f'data-bin/ailabel/{field}', '--workers', '40', 
        '--dataset-impl=lazy'
        ])


with Pool() as pool:
    pool.map(run, 
        configs.fields + [
            configs.static_field_label,
            # configs.var_label_field
            ]
    )

