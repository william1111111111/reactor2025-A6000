"""Frozen-weight alignment diagnostic on length-stratified VAL contexts."""
import argparse
import subprocess
import json
import time
from pathlib import Path
import numpy as np
import torch
from coreset_reactor.evaluate import _deterministic_postprocess, official_prediction
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.oracle_select import SharedQualityPool, Candidate
from coreset_reactor.teacher_cache import sha256_file
from framework.modules.post_processor import Processor
from framework.metrics import compute_s_mse, compute_FRVar
from regnn.conditional_model import ConditionalREGNNConfig
from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=int, default=6)
    parser.add_argument('--max-length', type=int, default=2250, help='0 includes all lengths')
    parser.add_argument('--output-name', default='direction_temporal_alignment_pilot6')
    args = parser.parse_args()
    if args.examples < 1 or args.max_length < 0 or Path(args.output_name).name != args.output_name:
        raise ValueError('Invalid experiment configuration')
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent / 'reports/fixed_teacher_distillation'
    source = root/'t10_student_direction_pilot10e/all_session_val571_full.json'
    ref = json.loads(source.read_text())
    cached = torch.load(source.with_suffix('.predictions.pt'), map_location='cpu')
    assert cached['checkpoint_sha256'] == ref['checkpoint_sha256'] == sha256_file(Path(ref['checkpoint']))
    assets = Path('/home/zhaowenjie/react2025_new/mam_reactor')
    data_root = Path('/public/zhaowenjie/react2025_new/data')
    data = PairedReactionDataset(data_root,'val',750,crop_mode='center',normalization_dir=assets/'external/FaceVerse')
    lengths = []
    excluded = []
    for i,p in enumerate(data.records):
        rel = p.relative_to(data.directory/'facial-attributes')
        lens = [len(np.load(data.directory/d/rel,mmap_mode='r')) for d in ('facial-attributes','audio-features','coefficients')]
        if len(set(lens)) != 1:
            excluded.append({'index': i, 'modality_lengths': lens})
        elif not args.max_length or lens[0] <= args.max_length:
            lengths.append((lens[0],i))
    lengths.sort()
    if args.examples > len(lengths):
        raise ValueError('Not enough eligible contexts')
    chosen = [lengths[i] for i in np.linspace(0,len(lengths)-1,args.examples,dtype=int)]
    output = root/args.output_name
    output.mkdir(exist_ok=False)
    provenance = {'arguments': vars(args), 'selected_length_and_index': chosen,
                  'eligible_contexts': len(lengths), 'excluded_modality_length_mismatch': excluded,
                  'checkpoint_sha256': ref['checkpoint_sha256'],
                  'script_sha256': sha256_file(Path(__file__)),
                  'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  'processor_sha256': sha256_file(Path(__file__).resolve().parents[1]/'mam_reactor/framework/modules/post_processor.py'),
                  'postprocessor_checkpoint_sha256': sha256_file(assets/'pretrained_models/post_processor/checkpoint.pth')}
    (output/'config.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance),flush=True)
    device=torch.device('cuda:5'); torch.cuda.set_device(device)
    state=torch.load(ref['checkpoint'],map_location='cpu'); c=state['student_config']
    model=ExpertConditionalREGNN(ExpertStudentConfig(ConditionalREGNNConfig(**c['backbone']),c['num_experts'],c['expert_embedding_dim'])).to(device).eval()
    model.load_state_dict(state['state_dict'])
    processor=Processor(cfg_dir=str(assets),ckpt_dir=str(assets/'pretrained_models/post_processor'),device=device,clip_len_test=1000,num_preds=10)
    max_gt=max(r['gt_count'] for r in ref['rows'])
    pool=SharedQualityPool(8,10,max(n for n,_ in chosen),max_targets=max_gt)
    rows=[]; started=time.monotonic()
    def score(pred,target):
        cc=[Candidate(pred[k],str(k),'direction',[('direction',k)]) for k in range(10)]
        frc,frd=pool.score(cc,target)
        return dict(FRC=float(frc.sum()),exact_FRD=float(frd.sum()),FRDiv=float(compute_s_mse([pred])),FRVar=float(compute_FRVar([pred])))
    try:
        with torch.inference_mode():
            for length,index in chosen:
                (output/'progress.json').write_text(json.dumps({'completed':len(rows),'total':len(chosen),
                                                              'active_index':index,'active_length':length}))
                sample=data[index]; row=ref['rows'][index]
                assert sample['clip_id']==row['clip_id']==cached['clip_ids'][index]
                rel=Path(sample['clip_id']+'.npy')
                streams=[torch.from_numpy(np.load(data.directory/d/rel).astype(np.float32)) for d in ('audio-features','facial-attributes','coefficients')]
                streams[2]=streams[2].reshape(length,58)
                streams[2]=(streams[2]-data.mean)/data.std
                parts=[]
                for start in range(0,length,750):
                    n=min(750,length-start)
                    chunk=[torch.nn.functional.pad(s[start:start+n],(0,0,0,750-n))[None].to(device) for s in streams]
                    parts.append(model.generate(*chunk,torch.tensor([n],device=device))['prediction'][0,:,:n].cpu())
                full=official_prediction(torch.cat(parts,dim=1))
                assert full.shape==(10,length,25) and torch.isfinite(full).all()
                crop=cached['official_predictions'][index]
                # Check fresh inference on the identical original source crop.
                n=int(sample['source_lengths'])
                fresh=official_prediction(model.generate(*[sample[k][None,:n].to(device) for k in ('speaker_audio','speaker_emotion','speaker_3dmm')],torch.tensor([n],device=device))['prediction'][0].cpu())
                torch.testing.assert_close(fresh,crop,atol=1e-5,rtol=1e-5)
                folder=data.directory/'facial-attributes/listener'/sample['session_id']
                paired=folder/rel.name
                paths=[paired]+[p for p in sorted(folder.glob('*.npy')) if p!=paired]
                assert len(paths)==row['gt_count']
                raw=[torch.from_numpy(np.load(p).astype(np.float32)) for p in paths]
                gt=_deterministic_postprocess(processor,full,raw,device,1234,sample['clip_id']).cpu()
                assert gt.shape==(len(paths),length,25) and torch.isfinite(gt).all()
                start=int(sample['crop_start'])
                aligned=gt[:,start:start+n]
                result={'index':index,'length':length,'crop_start':start,'GT_count':len(paths),
                        'legacy_crop':{'FRC':sum(row['frc']),'exact_FRD':row['exact_FRD'],'FRDiv':row['FRDiv']},
                        'same_prediction_aligned_GT':score(crop,aligned),
                        'full_prediction_full_GT':score(full,gt)}
                rows.append(result)
                torch.save({'full_prediction':full,'full_gt':gt},output/f'context_{index}.pt')
                report={'completed':len(rows),'total':len(chosen),'finished':len(rows)==len(chosen),'checkpoint_sha256':ref['checkpoint_sha256'],
                        'scope':'Length-stratified VAL pilot; all-session GT; not full VAL or official TEST. Full-first GT crop is a diagnostic convention. Full-length FRD is not comparable with crop FRD.',
                        'means':{arm:{metric:float(np.mean([r[arm][metric] for r in rows]))
                                      for metric in ('FRC','exact_FRD','FRDiv')}
                                 for arm in ('legacy_crop','same_prediction_aligned_GT','full_prediction_full_GT')},
                        'rows':rows,'elapsed_seconds':time.monotonic()-started}
                temp=output/'summary.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(output/'summary.json')
                print(json.dumps(result),flush=True)
    finally:
        pool.close()
    assert sha256_file(Path(ref['checkpoint']))==ref['checkpoint_sha256']


if __name__=='__main__':
    main()
