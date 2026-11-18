"""Independent post-run recomputation of costs, selection and holdout metrics."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

root = Path(__file__).resolve().parent
start = time.perf_counter()
p = json.loads((root/'parameters.json').read_text(encoding='utf-8'))
run = json.loads((root/'run.json').read_text(encoding='utf-8'))
assert run['code_sha256'] == hashlib.sha256((root/'calculate.py').read_bytes()).hexdigest()
assert run['parameters_sha256'] == hashlib.sha256((root/'parameters.json').read_bytes()).hexdigest()
with (root/'seasonal-results.csv').open(encoding='utf-8-sig',newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows)==1530
assert len({(r['point'],r['year'],r['profile'],r['variant']) for r in rows})==len(rows)
points = {}
for r in rows:
    pv, bat, litres = float(r['pv_kw']),float(r['usable_battery_kwh']),float(r['litres'])
    assert int(r['unknown_fuel_hours'])==0
    assert float(r['initial_soc_kwh'])==0
    assert 0 <= float(r['final_soc_kwh']) <= bat
    assert 0 <= float(r['max_balance_error_kwh']) < 1e-8
    cost = 80*litres + 4000*pv + 1500*bat
    assert abs(cost-float(r['seasonal_cost_rub'])) < 1e-6
    year = int(r['year'])
    assert (r['split']=='train') == (year<=2017)
    assert p['profiles'][r['profile']]['set']==r['split']
    if pv==0 and bat==0 and r['profile'] in ('flat','day','evening'):
        assert abs(litres-3672*3.25)<1e-6
    points.setdefault(r['point'],{}).setdefault((year,r['profile']),{})[r['variant']]=cost
summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
with (root/'holdout.csv').open(encoding='utf-8-sig',newline='') as f:
    held_rows = list(csv.DictReader(f))
assert len(held_rows)==36
for s in summary:
    scenarios = points[s['point']]
    training = [v for (year,profile),v in scenarios.items() if year<=2017]
    variants = [f'pv{pv}_bat{bat}' for pv,bat in p['variants']]
    scores = {v:max(cost[v]-min(cost.values()) for cost in training) for v in variants}
    robust = min(variants,key=lambda v:scores[v])
    baseline = min(variants,key=lambda v:scenarios[(2017,'flat')][v])
    assert robust==s['robust_variant'] and baseline==s['baseline_variant']
    assert abs(scores[robust]-s['training_worst_regret_rub'])<1e-6
    expected_status = 'no_clear_choice' if scores[robust]>p['user_max_regret_rub'] else 'within_assumed_threshold'
    assert s['selection_status']==expected_status
    point_holdout = [r for r in held_rows if r['point']==s['point']]
    for method,variant in [('robust',robust),('baseline',baseline)]:
        regrets=[]
        for r in point_holdout:
            costs=scenarios[(int(r['year']),r['profile'])]
            loss=costs[variant]-min(costs.values())
            assert abs(loss-float(r[method+'_regret_rub']))<1e-6
            regrets.append(loss)
        assert abs(statistics.mean(regrets)-s['holdout_'+method+'_mean_regret_rub'])<1e-6
        assert abs(max(regrets)-s['holdout_'+method+'_max_regret_rub'])<1e-6
result=dict(passed=True,seasonal_rows=len(rows),holdout_rows=len(held_rows),
            max_balance_error_kwh=max(float(r['max_balance_error_kwh']) for r in rows),
            verification_seconds=time.perf_counter()-start,
            checks=['unique complete rows','cost recomputation','data split','analytical diesel-only baseline',
                    'energy and SOC bounds','independent minimax and baseline choice','independent holdout losses','code and input hashes'])
(root/'independent-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
