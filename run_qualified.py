"""Run one explicitly qualified SuS Bayesian configuration."""
import argparse,json
from pathlib import Path
def describe(case):
    summary=json.loads((Path(__file__).parent/'VALIDATION_SUMMARY.json').read_text())
    row=next((r for r in summary['statistics'] if r['case_id']==case),None)
    if row is None:raise ValueError('Choose a listed qualified case')
    return dict(row,profile='closed_ties_unit_fallback_v1',estimator='source_native_evidence')
def construct(case):
    import sus_bayes as package
    cfg=describe(case)
    if case=='sus_bayes_eggbox2':target=package.LKF_2D_Eggbox()
    elif case=='sus_bayes_shells5':target=package.LKF_nD_GaussianShells(5)
    else:target=package.LKF_nD_GaussianLogGamma(2 if case.endswith('2') else 5)
    model=package.SuS_Bay_Nataf(target,method_profile=cfg['profile'])
    sampler=package.aCS(model);simulation=package.Main(sampler)
    simulation.NumSam=cfg['N'];simulation.MaxIte=10000;simulation.ConPrb=.1
    return cfg,model,sampler,simulation
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',required=True);parser.add_argument('--seed',type=int,default=1);parser.add_argument('--describe',action='store_true');args=parser.parse_args()
    if args.describe:print(json.dumps(describe(args.case),indent=2));return
    import numpy as np
    import sus_bayes as package
    np.random.seed(args.seed);cfg,model,sampler,simulation=construct(args.case)
    raw=simulation.RunIte();post=package.NumSim(raw)
    print(json.dumps(dict(case=args.case,seed=args.seed,profile=cfg['profile'],estimator=cfg['estimator'],logZ=np.asarray(post.Z).tolist(),Ncal=int(np.sum(raw['Ncal'])))))
if __name__=='__main__':main()
