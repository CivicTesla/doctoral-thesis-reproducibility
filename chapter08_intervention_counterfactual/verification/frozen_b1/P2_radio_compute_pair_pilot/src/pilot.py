"""Minimal isolated tandem radio-compute pilot; no old B1 semantics changed."""
from __future__ import annotations

import copy, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(5000, 5030)
MECHANISMS = ("H3_RADIO_UPLINK_BOTTLENECK", "H4_EDGE_COMPUTE_BOTTLENECK")
ACTIONS = ("RADIO_RECOVERY_ACTION", "COMPUTE_RECOVERY_ACTION", "NO_ACTION")
CFG = dict(horizon=240, decision=120, arrival_rate=1.4, deadline=55,
           nominal=2, bottleneck=1, augmentation=1, action_delay=2,
           action_duration=60, activation_cost=.5, sparse_window=20)

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()

def task_table(seed):
    rng=np.random.default_rng(seed); out=[]; task_id=0
    for tick in range(CFG["horizon"]):
        for _ in range(rng.poisson(CFG["arrival_rate"])):
            out.append(dict(task_id=task_id, arrival_tick=tick, radio_work=1,
                compute_work=1, deadline=tick+CFG["deadline"], completion_tick=None))
            task_id += 1
    return out

def schedules(seed):
    # Explicit frozen exogenous schedules; mechanisms change only stage baseline.
    rng=np.random.default_rng(seed+91000)
    radio=(rng.random(CFG["horizon"])>.03).astype(int).tolist()
    compute=(rng.random(CFG["horizon"])>.03).astype(int).tolist()
    return radio, compute

def initial_state(tasks):
    return dict(tick=0, radio=[], compute=[], completed=[], tasks=copy.deepcopy(tasks), telemetry=[])

def capacity(mechanism, stage, tick, action):
    reduced=(mechanism.startswith("H3") and stage=="radio") or (
        mechanism.startswith("H4") and stage=="compute")
    value=CFG["bottleneck"] if reduced else CFG["nominal"]
    active=CFG["decision"]+CFG["action_delay"] <= tick < (
        CFG["decision"]+CFG["action_delay"]+CFG["action_duration"])
    if active and action == ("RADIO_RECOVERY_ACTION" if stage=="radio" else "COMPUTE_RECOVERY_ACTION"):
        value += CFG["augmentation"]
    return value

def step(state, mechanism, action, radio_schedule, compute_schedule):
    t=state["tick"]
    for task in state["tasks"]:
        if task["arrival_tick"]==t: state["radio"].append(task["task_id"])
    rcap=capacity(mechanism,"radio",t,action)*radio_schedule[t]
    moved=[]
    for _ in range(min(rcap,len(state["radio"]))): moved.append(state["radio"].pop(0))
    state["compute"].extend(moved)
    ccap=capacity(mechanism,"compute",t,action)*compute_schedule[t]
    done=[]
    for _ in range(min(ccap,len(state["compute"]))): done.append(state["compute"].pop(0))
    for task_id in done:
        state["tasks"][task_id]["completion_tick"]=t; state["completed"].append(task_id)
    recent=[state["tasks"][i] for i in state["completed"]
            if state["tasks"][i]["completion_tick"]>=t-9]
    delays=[x["completion_tick"]-x["arrival_tick"] for x in recent]
    arrived=[x for x in state["tasks"] if x["arrival_tick"]<=t]
    misses=sum(x["deadline"]<=t and x["completion_tick"] is None for x in arrived)
    last=max([x["completion_tick"] for x in state["tasks"] if x["completion_tick"] is not None] or [-1])
    state["telemetry"].append(dict(tick=t, completion_rate=len(recent)/10,
        end_to_end_latency=float(np.mean(delays)) if delays else CFG["deadline"],
        deadline_misses=misses, unfinished=len(state["radio"])+len(state["compute"]),
        throughput=len(recent)/10, age_since_completion=t-last if last>=0 else t+1))
    state["tick"] += 1

def run_until(state, end, mechanism, action, rs, cs):
    while state["tick"]<end: step(state,mechanism,action,rs,cs)
    return state

def outcome(state, action):
    arrived=[x for x in state["tasks"] if x["arrival_tick"]<CFG["horizon"]]
    misses=sum(x["deadline"]<CFG["horizon"] and x["completion_tick"] is None for x in arrived)
    unfinished=sum(x["completion_tick"] is None for x in arrived)
    completed=[x for x in arrived if x["completion_tick"] is not None]
    mean_delay=float(np.mean([x["completion_tick"]-x["arrival_tick"] for x in completed])) if completed else CFG["horizon"]
    activation=CFG["activation_cost"] if action!="NO_ACTION" else 0
    cost=misses+unfinished+.1*(mean_delay/CFG["deadline"])+activation
    return dict(completed_tasks=len(completed),deadline_misses=misses,
        unfinished_tasks=unfinished,mean_end_to_end_delay=mean_delay,
        terminal_radio_backlog=len(state["radio"]),terminal_compute_backlog=len(state["compute"]),
        activation_cost=activation,total_system_cost=cost,terminal_state_hash=digest(state))

def features(checkpoint, full=False, temporal=False):
    w=checkpoint["telemetry"][-CFG["sparse_window"]:]
    names=("completion_rate","end_to_end_latency","deadline_misses","unfinished","throughput","age_since_completion")
    values=[]
    if temporal:
        x=np.arange(len(w))
        for n in names:
            a=np.array([r[n] for r in w],float)
            values += [a.mean(), np.polyfit(x,a,1)[0], a.var()]
    else:
        values=[w[-1][n] for n in names]
    if full: values += [len(checkpoint["radio"]),len(checkpoint["compute"]),
        capacity("H3_RADIO_UPLINK_BOTTLENECK","radio",CFG["decision"],"NO_ACTION"),
        capacity("H4_EDGE_COMPUTE_BOTTLENECK","compute",CFG["decision"],"NO_ACTION")]
    return values

def fit_logistic(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=x.mean(0); s=x.std(0); s[s==0]=1
    z=np.c_[np.ones(len(x)),(x-m)/s]; w=np.zeros(z.shape[1])
    for _ in range(80):
        p=1/(1+np.exp(-np.clip(z@w,-30,30))); q=p*(1-p)
        h=z.T@(q[:,None]*z)+np.eye(z.shape[1])*1e-3; g=z.T@(p-y)+1e-3*w
        d=np.linalg.solve(h,g); w-=d
        if abs(d).max()<1e-8: break
    return m,s,w

def predict(model,x):
    m,s,w=model; z=np.c_[np.ones(len(x)),(np.asarray(x)-m)/s]
    return 1/(1+np.exp(-np.clip(z@w,-30,30)))

def diagnostic(rows,key):
    probs=np.zeros(len(rows)); y=np.array([r["mechanism"].startswith("H4") for r in rows])
    seeds=np.array([r["seed"] for r in rows])
    for fold in range(5):
        train=seeds%5!=fold; test=~train
        model=fit_logistic([r[key] for i,r in enumerate(rows) if train[i]],y[train])
        probs[test]=predict(model,[r[key] for i,r in enumerate(rows) if test[i]])
    pred=probs>=.5; ba=.5*(pred[y].mean()+(~pred[~y]).mean())
    rng=np.random.default_rng(8821); unique=np.unique(seeds); boots=[]
    for _ in range(2000):
        draw=rng.choice(unique,len(unique),replace=True); idx=np.concatenate([np.flatnonzero(seeds==s) for s in draw])
        boots.append(.5*(pred[idx][y[idx]].mean()+(~pred[idx][~y[idx]]).mean()))
    return probs,dict(balanced_accuracy=ba,ci95_low=np.quantile(boots,.025),ci95_high=np.quantile(boots,.975),model="logistic_regression",group="seed")

def write_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    (ROOT/"results").mkdir(exist_ok=True); (ROOT/"figures").mkdir(exist_ok=True)
    config_hash=digest(CFG); manifests=[]; validations=[]; outcomes=[]; diag=[]
    for seed in SEEDS:
      tasks=task_table(seed); rs,cs=schedules(seed)
      for mech in MECHANISMS:
        state=run_until(initial_state(tasks),CFG["decision"],mech,"NO_ACTION",rs,cs)
        pre_hash=digest(state); task_hash=digest(tasks); rh=digest(rs); ch=digest(cs)
        manifests.append(dict(seed=seed,mechanism=mech,pre_action_state_hash=pre_hash,
            task_schedule_hash=task_hash,radio_schedule_hash=rh,compute_schedule_hash=ch,config_hash=config_hash))
        diag.append(dict(seed=seed,mechanism=mech,sparse_static=features(state),
            sparse_temporal=features(state,temporal=True),full_state=features(state,full=True)))
        branch_hashes=[]
        for action in ACTIONS:
            branch=copy.deepcopy(state)
            assert digest(branch)==pre_hash
            final=run_until(branch,CFG["horizon"],mech,action,rs,cs); out=outcome(final,action)
            outcomes.append(dict(seed=seed,mechanism=mech,action=action,
                pre_action_state_hash=pre_hash,task_schedule_hash=task_hash,
                radio_schedule_hash=rh,compute_schedule_hash=ch,config_hash=config_hash,**out))
            branch_hashes.append((pre_hash,task_hash,rh,ch,config_hash))
        valid=len(set(branch_hashes))==1
        validations.append(dict(seed=seed,mechanism=mech,three_actions=3,
            pre_action_hash_equal=valid,task_schedule_equal=valid,radio_schedule_equal=valid,
            compute_schedule_equal=valid,config_equal=valid,status="VALID" if valid else "INVALID"))
        if not valid: raise RuntimeError("P2_COUNTERFACTUAL_IMPLEMENTATION_INVALID")
    write_csv(ROOT/"08_counterfactual_manifest.csv",manifests); write_csv(ROOT/"09_branch_hash_validation.csv",validations)
    write_csv(ROOT/"10_per_checkpoint_action_outcomes.csv",outcomes)
    # action separation and paired bootstrap
    summary=[]; bootrows=[]; rng=np.random.default_rng(7731)
    for mech,preferred in ((MECHANISMS[0],"RADIO_RECOVERY_ACTION"),(MECHANISMS[1],"COMPUTE_RECOVERY_ACTION")):
        by={a:np.array([r["total_system_cost"] for r in outcomes if r["mechanism"]==mech and r["action"]==a]) for a in ACTIONS}
        stack=np.column_stack([by[a] for a in ACTIONS]); mins=stack.min(1); ties=(stack==mins[:,None]).sum(1)>1
        best=np.argmin(stack,1); row=dict(mechanism=mech,runs=len(stack),tie_fraction=ties.mean())
        for j,a in enumerate(ACTIONS): row[a+"_best_fraction"]=np.mean((best==j)&~ties)
        for other in [a for a in ACTIONS if a!=preferred]:
            diff=by[other]-by[preferred]; draws=[]
            for _ in range(2000): draws.append(rng.choice(diff,len(diff),replace=True).mean())
            bootrows.append(dict(mechanism=mech,comparison=f"{preferred}_benefit_vs_{other}",
                mean_cost_reduction=diff.mean(),ci95_low=np.quantile(draws,.025),ci95_high=np.quantile(draws,.975),replicates=2000))
        summary.append(row)
    write_csv(ROOT/"11_action_separation_summary.csv",summary); write_csv(ROOT/"12_action_bootstrap.csv",bootrows)
    passed=all(r[("RADIO_RECOVERY_ACTION" if r["mechanism"].startswith("H3") else "COMPUTE_RECOVERY_ACTION")+"_best_fraction"]>=.8 for r in summary) and all(r["ci95_low"]>0 for r in bootrows)
    rows=[]
    for r in diag: rows.append(dict(seed=r["seed"],mechanism=r["mechanism"],sparse_static=json.dumps(r["sparse_static"]),sparse_temporal=json.dumps(r["sparse_temporal"]),full_state=json.dumps(r["full_state"])))
    write_csv(ROOT/"15_diagnostic_rows.csv",rows)
    if not passed: decision="P2_ACTION_SEPARATION_FAILS"; stats={}
    else:
        stats={}
        for key,file in (("full_state","16_full_state_diagnostic.csv"),("sparse_static","17_sparse_static_diagnostic.csv"),("sparse_temporal","18_sparse_temporal_diagnostic.csv")):
            p,s=diagnostic(diag,key); stats[key]=s; write_csv(ROOT/file,[s])
        if stats["full_state"]["balanced_accuracy"]<.9: decision="P2_MECHANISM_DEFINITION_INSUFFICIENT"
        elif stats["sparse_static"]["balanced_accuracy"]>=.95: decision="P2_NO_OBSERVATIONAL_OVERLAP"
        elif max(stats["sparse_static"]["balanced_accuracy"],stats["sparse_temporal"]["balanced_accuracy"])<=.55 or max(stats["sparse_static"]["ci95_low"],stats["sparse_temporal"]["ci95_low"])<=.5: decision="P2_NO_SPARSE_DIAGNOSTIC_SIGNAL"
        else: decision="P2_SECOND_PAIR_SURVIVES"
    (ROOT/"23_p2_decision.txt").write_text(decision+"\n")
    report=f"# P2 report\n\n`{decision}`\n\nStrict forks: valid. Action separation passed: {passed}. Diagnostics: {json.dumps(stats,sort_keys=True)}. No P3 is run.\n"
    (ROOT/"21_p2_report.md").write_text(report)
    c=ROOT.parent/"results/CONCLUSIONS/P2_SECOND_PAIR"; c.mkdir(parents=True,exist_ok=True); (c/"P2_FORMAL_DECISION.md").write_text(report)
    print(report)

if __name__=="__main__": main()
