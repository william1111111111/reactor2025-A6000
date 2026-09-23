"""Three disjoint certified-oracle shards and an atomic full-VAL progress report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from coreset_reactor.run_certified_oracle import write_report, atomic_json, ROOT
from coreset_reactor.teacher_cache import sha256_file


def merge_shard_records(summaries):
    reference=summaries[0]['config']
    records={}
    assigned=set()
    for summary in summaries:
        config=summary['config']
        for key in ('checkpoint_sources','source_hashes','eval_seed','metric_workers','mip_seconds',
                    'primary_pool','analysis_pool','quality_policy','population_contexts','reuse_dir'):
            if config[key]!=reference[key]:
                raise ValueError('Shard protocol mismatch: '+key)
        indices=set(config['indices'])
        if assigned & indices:
            raise ValueError('Overlapping shard assignments')
        assigned.update(indices)
        expected=dict(zip(config['indices'],config['clip_ids']))
        for row in summary['per_context']:
            index=row['val_index']
            if index in records or expected.get(index)!=row['clip_id']:
                raise ValueError('Duplicate/misassigned context')
            records[index]=row
    return [records[k] for k in sorted(records)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-dir',type=Path,required=True)
    parser.add_argument('--shards',type=int,default=3)
    parser.add_argument('--workers-per-shard',type=int,default=8)
    parser.add_argument('--mip-seconds',type=float,default=5.)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not 1<=args.shards<=8:
        raise ValueError('Use 1..8 bounded shards')
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    edges=np.linspace(0,571,args.shards+1,dtype=int).tolist()
    launch={'shards':args.shards,'edges':edges,'workers_per_shard':args.workers_per_shard,
            'mip_seconds':args.mip_seconds,'reuse_dir':str(args.reuse_dir.resolve()),
            'coordinator_sha256':sha256_file(Path(__file__)),
            'runner_sha256':sha256_file(ROOT/'coreset_reactor/run_certified_oracle.py')}
    config_file=args.output/'launch.json'
    if config_file.exists() and json.loads(config_file.read_text())!=launch:
        raise ValueError('Resume launch mismatch')
    atomic_json(config_file,launch)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
         'PYTHONPATH':str(ROOT/'mam_reactor')+os.pathsep+str(ROOT)}
    jobs=[]
    started=time.monotonic()
    for shard,(start,stop) in enumerate(zip(edges[:-1],edges[1:])):
        directory=args.output/f'shard_{shard}'
        cmd=[sys.executable,'-u','-m','coreset_reactor.run_certified_oracle',
             '--output',str(directory),'--contexts','571','--context-start',str(start),
             '--context-stop',str(stop),'--workers',str(args.workers_per_shard),
             '--mip-seconds',str(args.mip_seconds),'--reuse-dir',str(args.reuse_dir)]
        if args.resume and directory.exists():
            cmd.append('--resume')
        log=(args.output/f'shard_{shard}.log').open('a')
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((proc,log,directory,cmd))
    last=-1
    while True:
        summaries=[]
        for _,_,directory,_ in jobs:
            path=directory/'summary.json'
            if path.exists():
                summaries.append(json.loads(path.read_text()))
        statuses=[{'pid':p.pid,'returncode':p.poll(),'directory':str(d),'command':cmd} for p,_,d,cmd in jobs]
        records=merge_shard_records(summaries) if summaries else []
        if summaries and len(records)!=last:
            reference=summaries[0]['config']
            all_clips={i:clip for s in summaries for i,clip in zip(s['config']['indices'],s['config']['clip_ids'])}
            global_config={**reference,'contexts':571,'population_contexts':571,
                           'context_start':0,'context_stop':571,'indices':list(range(571)),
                           'clip_ids':[all_clips.get(i) for i in range(571)],
                           'shards':args.shards,'workers_total':args.workers_per_shard*args.shards,
                           'report_scope':'partial until all 571 contexts complete'}
            write_report(args.output,global_config,records)
            last=len(records)
            print(json.dumps({'completed':last,'total':571,
                              'reused':sum('reused_from' in r for r in records),
                              'elapsed_seconds':time.monotonic()-started}),flush=True)
        finished=all(s['returncode'] is not None for s in statuses)
        failed=[s for s in statuses if s['returncode'] not in (0,None)]
        atomic_json(args.output/'status.json',{'completed':len(records),'total':571,
                    'reused':sum('reused_from' in r for r in records),'finished':finished,
                    'elapsed_seconds':time.monotonic()-started,'jobs':statuses,'failed_jobs':failed})
        if finished:
            for _,log,_,_ in jobs:
                log.close()
            if failed or len(records)!=571:
                raise RuntimeError(f'Full evaluation incomplete: {len(records)}/571, {len(failed)} failed shards')
            print('Full VAL571 complete; no missing/duplicate contexts.',flush=True)
            return
        time.sleep(20)


if __name__=='__main__':
    main()
