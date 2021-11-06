from commands.configs import fields, static_field_label
from multiprocessing import Pool
import subprocess


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/mask_ail_trace_dicts/{field}/dict.txt', '--trainpref',
         f'data-src/mask_ail_trace/train.{field}',
         '--validpref',
         f'data-src/mask_ail_trace/valid.{field}', '--destdir', f'data-bin/mask_ail_trace/{field}', '--workers',
         '40'])


with Pool() as pool:
    pool.map(run, fields+[static_field_label])
