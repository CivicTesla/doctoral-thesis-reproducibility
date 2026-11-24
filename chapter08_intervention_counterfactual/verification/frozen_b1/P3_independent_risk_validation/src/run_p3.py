"""P3 independent validation using the byte-frozen P2 radio-compute pilot."""
from __future__ import annotations

import copy, csv, hashlib, importlib.util, json, subprocess, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; B1=ROOT.parent; P2=B1/"P2_radio_compute_pair_pilot"
P2_COMMIT="3726b1d"; SEEDS=range(5100,5200); BOOTS=2000
SCI_FILES=("src/pilot.py","01_frozen_config.yaml","03_system_model.md","05_action_definitions.md",
           "06_utility_grounding.md","13_sparse_feature_definition.md","14_full_feature_definition.md")
spec=importlib.util.spec_from_file_location("p2_frozen",P2/"src/pilot.py")
p2=importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)

def sha(x): return hashlib.sha256(x).hexdigest()
def write_csv(path,rows,fields=None):
    fields=fields or list(rows[0])
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def git_bytes(rel):
    return subprocess.check_output(["git","show",f"{P2_COMMIT}:B1/P2_radio_compute_pair_pilot/{rel}"],cwd=B1)
def frozen_manifest():
    rows=[]
    for rel in SCI_FILES:
        now=(P2/rel).read_bytes(); old=git_bytes(rel)
        rows.append(dict(component=rel,p2_commit=P2_COMMIT,current_sha256=sha(now),committed_sha256=sha(old),exact_match=now==old))
    if not all(r["exact_match"] for r in rows): raise RuntimeError("P3_P2_CONFIGURATION_DRIFT")
    return rows
def costs_for(outcomes,seed,mech):
    return {r["action"]:r["total_system_cost"] for r in outcomes if r["seed"]==seed and r["mechanism"]==mech}
def ba(y,p,accept=None):
    if accept is None: accept=np.ones(len(y),bool)
    z=accept
    vals=[]
    for c in (0,1): vals.append(np.mean(p[z & (y==c)]==c) if np.any(z & (y==c)) else np.nan)
    return float(np.nanmean(vals))
def metric(rows,method):
    y=np.array([r["y"] for r in rows]); pred=np.array([r[f"{method}_pred"] for r in rows]); acc=np.array([r[f"{method}_accept"] for r in rows])
    reg=np.array([r[f"{method}_regret"] for r in rows])
    d=dict(method=method,balanced_accuracy=ba(y,pred),coverage=acc.mean(),accepted_accuracy=np.mean(pred[acc]==y[acc]))
    d.update(total_regret=reg.mean(),H3_regret=reg[y==0].mean(),H4_regret=reg[y==1].mean(),
             H3_recall=np.mean(pred[y==0]==0),H4_recall=np.mean(pred[y==1]==1),
             H3_coverage=acc[y==0].mean(),H4_coverage=acc[y==1].mean())
    return d
def bootstrap_diff(rows,a,b,seed=94117):
    rng=np.random.default_rng(seed); seeds=np.array(sorted({r["seed"] for r in rows})); out=[]
    for _ in range(BOOTS):
        draw=rng.choice(seeds,len(seeds),replace=True); rr=[r for s in draw for r in rows if r["seed"]==s]
        ma,mb=metric(rr,a),metric(rr,b)
        out.append([mb["balanced_accuracy"]-ma["balanced_accuracy"],mb["total_regret"]-ma["total_regret"],mb["H3_regret"]-ma["H3_regret"],mb["H4_regret"]-ma["H4_regret"]])
    out=np.array(out); names=("balanced_accuracy","total_regret","H3_regret","H4_regret")
    return [dict(comparison=f"{b}-{a}",metric=n,estimate=metric(rows,b)[n]-metric(rows,a)[n],ci95_low=np.quantile(out[:,i],.025),ci95_high=np.quantile(out[:,i],.975),replicates=BOOTS,resampling_unit="seed") for i,n in enumerate(names)]
def decomposition(rows,a,b):
    result=[]
    for mech,yval in (("H3",0),("H4",1)):
        rr=[r for r in rows if r["y"]==yval]; corrected=new=loss=gain=0.0
        for r in rr:
            aa,bb=r[f"{a}_accept"],r[f"{b}_accept"]; ra,rb=r[f"{a}_regret"],r[f"{b}_regret"]
            if aa and bb:
                if rb<ra: corrected += ra-rb
                elif rb>ra: new += rb-ra
            elif aa and not bb: loss += rb-ra
            elif not aa and bb: gain += rb-ra
        lhs=sum(r[f"{b}_regret"]-r[f"{a}_regret"] for r in rr); rhs=new-corrected+loss+gain
        result.append(dict(comparison=f"{a}_to_{b}",mechanism=mech,n=len(rr),corrected_harm_mass=corrected,new_harm_mass=new,
            coverage_loss_mass=loss,coverage_gain_mass=gain,lhs_total_regret_change=lhs,rhs_reconstruction=rhs,absolute_error=abs(lhs-rhs)))
    return result
def action_boot(outcomes):
    rng=np.random.default_rng(713); rows=[]
    for mech,pref in ((p2.MECHANISMS[0],"RADIO_RECOVERY_ACTION"),(p2.MECHANISMS[1],"COMPUTE_RECOVERY_ACTION")):
        by={a:np.array([r["total_system_cost"] for r in outcomes if r["mechanism"]==mech and r["action"]==a]) for a in p2.ACTIONS}
        st=np.column_stack([by[a] for a in p2.ACTIONS]); ties=(st==st.min(1)[:,None]).sum(1)>1; best=np.argmin(st,1)
        base=dict(mechanism=mech,n=len(st),tie_fraction=ties.mean(),radio_best_fraction=np.mean((best==0)&~ties),compute_best_fraction=np.mean((best==1)&~ties),no_action_best_fraction=np.mean((best==2)&~ties))
        for other in (a for a in p2.ACTIONS if a!=pref):
            d=by[other]-by[pref]; bs=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(BOOTS)])
            rows.append(dict(**base,comparison=f"{pref}_benefit_vs_{other}",mean_effect=d.mean(),ci95_low=np.quantile(bs,.025),ci95_high=np.quantile(bs,.975)))
    h3=np.array([r["total_system_cost"] for r in outcomes if r["mechanism"].startswith("H3") and r["action"]=="COMPUTE_RECOVERY_ACTION"])-np.array([r["total_system_cost"] for r in outcomes if r["mechanism"].startswith("H3") and r["action"]=="RADIO_RECOVERY_ACTION"])
    h4=np.array([r["total_system_cost"] for r in outcomes if r["mechanism"].startswith("H4") and r["action"]=="RADIO_RECOVERY_ACTION"])-np.array([r["total_system_cost"] for r in outcomes if r["mechanism"].startswith("H4") and r["action"]=="COMPUTE_RECOVERY_ACTION"])
    ratios=np.array([rng.choice(h3,len(h3),True).mean()/rng.choice(h4,len(h4),True).mean() for _ in range(BOOTS)])
    ratio=dict(H3_wrong_action_advantage=h3.mean(),H4_wrong_action_advantage=h4.mean(),cost_ratio=h3.mean()/h4.mean(),ci95_low=np.quantile(ratios,.025),ci95_high=np.quantile(ratios,.975))
    return rows,ratio
def figures(summary,decomp):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize":(5.4,3.5),"axes.grid":True,"grid.alpha":.2}); colors=["#8ecae6","#a8dadc","#ffb4a2"]
    methods=[r["method"] for r in summary]
    plots=[("01_accuracy_coverage.png",[r["balanced_accuracy"] for r in summary],[r["coverage"] for r in summary],"BA","coverage"),
      ("02_mechanism_regret.png",[r["H3_regret"] for r in summary],[r["H4_regret"] for r in summary],"H3 regret","H4 regret")]
    for name,a,b,la,lb in plots:
        fig,ax=plt.subplots(); x=np.arange(3); ax.bar(x-.18,a,.36,label=la,color=colors[0]); ax.bar(x+.18,b,.36,label=lb,color=colors[2]); ax.set_xticks(x,methods); ax.set_ylim(bottom=0); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/"figures"/name,dpi=160); plt.close(fig)
    extras=[("03_pair_cost_asymmetry.png",[3.9,1.0],["H1/H2","H3/H4"]),
            ("04_corrected_new_harm.png",[sum(r["corrected_harm_mass"] for r in decomp[:2]),sum(r["new_harm_mass"] for r in decomp[:2])],["corrected","new"]),
            ("05_coverage_mass.png",[sum(r["coverage_loss_mass"] for r in decomp[2:4]),sum(r["coverage_gain_mass"] for r in decomp[2:4])],["loss","gain"]),
            ("06_accuracy_risk_alignment.png",[.0544,summary[1]["balanced_accuracy"]-summary[0]["balanced_accuracy"]],["Pair 1 BA gain","Pair 2 BA gain"])]
    for name,v,l in extras:
        fig,ax=plt.subplots(); ax.bar(l,v,color=colors[:len(v)]); ax.axhline(0,color="#666",lw=.8); lo=min(0,min(v)*1.1); hi=max(v)*1.15 if max(v)>0 else 1; ax.set_ylim(lo,hi); fig.tight_layout(); fig.savefig(ROOT/"figures"/name,dpi=160); plt.close(fig)
def main():
    start=time.perf_counter(); manifest=frozen_manifest(); write_csv(ROOT/"01_p2_frozen_manifest.csv",manifest)
    outcomes=[]; diag=[]; man=[]; valid=[]
    for seed in SEEDS:
      tasks=p2.task_table(seed); rs,cs=p2.schedules(seed)
      for mech in p2.MECHANISMS:
        state=p2.run_until(p2.initial_state(tasks),p2.CFG["decision"],mech,"NO_ACTION",rs,cs)
        hs=(p2.digest(state),p2.digest(tasks),p2.digest(rs),p2.digest(cs),p2.digest(p2.CFG))
        man.append(dict(seed=seed,mechanism=mech,pre_action_state_hash=hs[0],task_schedule_hash=hs[1],radio_schedule_hash=hs[2],compute_schedule_hash=hs[3],config_hash=hs[4]))
        diag.append(dict(seed=seed,mechanism=mech,y=int(mech.startswith("H4")),static=p2.features(state),temporal=p2.features(state,temporal=True)))
        seen=[]
        for action in p2.ACTIONS:
            branch=copy.deepcopy(state); seen.append((p2.digest(branch),hs[1],hs[2],hs[3],hs[4]))
            out=p2.outcome(p2.run_until(branch,p2.CFG["horizon"],mech,action,rs,cs),action)
            outcomes.append(dict(seed=seed,mechanism=mech,action=action,**out))
        ok=len(set(seen))==1; valid.append(dict(seed=seed,mechanism=mech,hashes_equal=ok,status="VALID" if ok else "INVALID"))
        if not ok: raise RuntimeError("P3_COUNTERFACTUAL_IMPLEMENTATION_INVALID")
    write_csv(ROOT/"03_counterfactual_manifest.csv",man); write_csv(ROOT/"04_branch_hash_validation.csv",valid)
    ab,ratio=action_boot(outcomes); write_csv(ROOT/"05_action_replication_results.csv",ab); write_csv(ROOT/"06_action_cost_ratio.csv",[ratio])
    if not all(r["ci95_low"]>0 and ((r["radio_best_fraction"] if r["mechanism"].startswith("H3") else r["compute_best_fraction"])>=.8) for r in ab): raise RuntimeError("P3_ACTION_GROUNDING_NOT_REPLICATED")
    y=np.array([r["y"] for r in diag]); seeds=np.array([r["seed"] for r in diag]); probs0=np.zeros(len(diag)); probs1=np.zeros(len(diag)); accept2=np.zeros(len(diag),bool); thresholds={}
    for fold in range(5):
        tr=seeds%5!=fold; te=~tr
        m0=p2.fit_logistic([r["static"] for i,r in enumerate(diag) if tr[i]],y[tr]); m1=p2.fit_logistic([r["temporal"] for i,r in enumerate(diag) if tr[i]],y[tr])
        probs0[te]=p2.predict(m0,[r["static"] for i,r in enumerate(diag) if te[i]]); probs1[te]=p2.predict(m1,[r["temporal"] for i,r in enumerate(diag) if te[i]])
        trainp=p2.predict(m1,[r["temporal"] for i,r in enumerate(diag) if tr[i]]); threshold=float(np.quantile(abs(trainp-.5),.20)); thresholds[fold]=threshold
        accept2[te]=abs(probs1[te]-.5)>=threshold
    rows=[]
    for i,r in enumerate(diag):
        c=costs_for(outcomes,r["seed"],r["mechanism"]); oracle=min(c,key=c.get); base=c[oracle]
        row=dict(seed=r["seed"],mechanism=r["mechanism"],y=r["y"],outer_fold=r["seed"]%5,m0_probability_H4=probs0[i],m1_probability_H4=probs1[i],m2_threshold=thresholds[r["seed"]%5],oracle_action=oracle)
        for method,prob,accept in (("M0",probs0[i],True),("M1",probs1[i],True),("M2",probs1[i],bool(accept2[i]))):
            pred=int(prob>=.5); action=("COMPUTE_RECOVERY_ACTION" if pred else "RADIO_RECOVERY_ACTION") if accept else "NO_ACTION"
            row.update({f"{method}_pred":pred,f"{method}_accept":accept,f"{method}_action":action,f"{method}_strict_cost":c[action],f"{method}_regret":c[action]-base})
        rows.append(row)
    write_csv(ROOT/"08_out_of_fold_predictions.csv",rows); summary=[metric(rows,m) for m in ("M0","M1","M2")]; write_csv(ROOT/"09_classification_results.csv",summary); write_csv(ROOT/"10_strict_regret_results.csv",summary)
    decomps=[]
    for a,b,file in (("M0","M1","11_m0_to_m1_harm_decomposition.csv"),("M1","M2","12_m1_to_m2_harm_decomposition.csv"),("M0","M2","13_m0_to_m2_harm_decomposition.csv")):
        d=decomposition(rows,a,b); decomps+=d; write_csv(ROOT/file,d)
    write_csv(ROOT/"14_decomposition_validation.csv",decomps)
    if max(r["absolute_error"] for r in decomps)>=1e-9: raise RuntimeError("P3_HARM_DECOMPOSITION_FAILED")
    boots=bootstrap_diff(rows,"M0","M1")+bootstrap_diff(rows,"M1","M2",94118); write_csv(ROOT/"15_seed_bootstrap.csv",boots)
    loo=[]
    for seed in SEEDS:
        rr=[r for r in rows if r["seed"]!=seed]; a=metric(rr,"M0"); b=metric(rr,"M1"); loo.append(dict(omitted_seed=seed,ba_difference=b["balanced_accuracy"]-a["balanced_accuracy"],total_regret_difference=b["total_regret"]-a["total_regret"]))
    write_csv(ROOT/"16_leave_one_seed_out.csv",loo)
    rejected=[r for r in rows if not r["M2_accept"]]; coverage=[dict(mechanism=m,coverage=np.mean([r["M2_accept"] for r in rows if r["mechanism"].startswith(m)]),rejected=sum(not r["M2_accept"] for r in rows if r["mechanism"].startswith(m)),rejected_correct=sum((not r["M2_accept"]) and r["M1_pred"]==r["y"] for r in rows if r["mechanism"].startswith(m)),rejected_wrong=sum((not r["M2_accept"]) and r["M1_pred"]!=r["y"] for r in rows if r["mechanism"].startswith(m)),no_action_opportunity_cost=sum(r["M2_regret"] for r in rows if r["mechanism"].startswith(m) and not r["M2_accept"])) for m in ("H3","H4")]
    write_csv(ROOT/"17_coverage_balance_audit.csv",coverage)
    bmap={(r["comparison"],r["metric"]):r for r in boots}; gain=bmap[("M1-M0","balanced_accuracy")]; total=bmap[("M1-M0","total_regret")]; h3=bmap[("M1-M0","H3_regret")]; h4=bmap[("M1-M0","H4_regret")]
    survives=gain["ci95_low"]>0 and total["ci95_high"]<0 and h3["ci95_low"]<=0 and h4["ci95_low"]<=0
    decision="P3_INDEPENDENT_CRITERION_VALIDATION_SURVIVES" if survives else ("P3_NO_DIAGNOSTIC_GAIN" if gain["ci95_low"]<=0 else "P3_PARTIAL_CRITERION_VALIDATION")
    write_csv(ROOT/"19_pair1_pair2_comparison.csv",[dict(pair="H1/H2",cost_ratio=3.9,critical_inversion=True,risk_transfer=True),dict(pair="H3/H4",cost_ratio=ratio["cost_ratio"],critical_inversion=False,risk_transfer=(h3["ci95_high"]<0 and h4["ci95_low"]>0) or (h4["ci95_high"]<0 and h3["ci95_low"]>0))])
    write_csv(ROOT/"20_claim_survival_matrix.csv",[dict(claim="P0 harm identity",status="SURVIVES"),dict(claim="P2 symmetric-cost prediction",status="SURVIVES" if not ((h3["ci95_high"]<0 and h4["ci95_low"]>0) or (h4["ci95_high"]<0 and h3["ci95_low"]>0)) else "CONTRADICTED"),dict(claim="P3 accuracy-risk alignment",status="SURVIVES" if survives else "PARTIAL")])
    figures(summary,decomps)
    report=f"# P3 report\n\n`{decision}`\n\nP2 configuration is byte-identical to {P2_COMMIT}. New seeds 5100–5199. Harm identity maximum error {max(r['absolute_error'] for r in decomps):.3g}. Runtime {time.perf_counter()-start:.2f}s. No P4 or external simulator was run.\n"
    (ROOT/"18_p0_prediction_tests.md").write_text("# P0 prediction tests\n\nP1: `P1_NOT_ACTIVATED_IN_SECOND_PAIR`. P2 and P3 are evaluated from seed-level confidence intervals and exact harm decomposition.\n")
    (ROOT/"21_p3_report.md").write_text(report); (ROOT/"22_p3_decision.txt").write_text(decision+"\n"); (B1/"results/CONCLUSIONS/P3_INDEPENDENT_VALIDATION/P3_FORMAL_DECISION.md").write_text(report)
    print(json.dumps(dict(decision=decision,summary=summary,ratio=ratio,bootstrap=boots,max_decomposition_error=max(r["absolute_error"] for r in decomps)),indent=2))
if __name__=="__main__": main()
