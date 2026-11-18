"""Independent CSV verification and price-of-error diagnostics for experiment 2."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'experiment2'
begin = time.perf_counter()
p = json.loads((OUT/'parameters.json').read_text(encoding='utf-8'))
run = json.loads((OUT/'run.json').read_text(encoding='utf-8'))
for file, key in [('experiment2.py', 'code_sha256'), ('calculate.py', 'model_sha256')]:
    assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == run[key]
assert hashlib.sha256((OUT/'parameters.json').read_bytes()).hexdigest() == run['parameters_sha256']
with (OUT/'seasonal-results.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == run['seasonal_runs']
assert len({(r['point'], r['year'], r['profile'], r['variant']) for r in rows}) == len(rows)
data = {}
fuel_costs = {}
for r in rows:
    key = (r['point'], int(r['year']), r['profile'])
    pv, bat, litres = float(r['pv_kw']), float(r['usable_battery_kwh']), float(r['litres'])
    assert int(r['unknown_fuel_hours']) == 0
    assert float(r['initial_soc_kwh']) == 0
    assert 0 <= float(r['final_soc_kwh']) <= bat
    assert float(r['max_balance_error_kwh']) < 1e-8
    cost = litres*p['fuel_rub_per_litre'] + pv*p['pv_season_rub_per_kw'] + bat*p['battery_season_rub_per_kwh']
    assert abs(cost-float(r['seasonal_cost_rub'])) < 1e-6
    assert p['profiles'][r['profile']]['set'] == ('train' if key[1] <= 2017 else 'holdout')
    if pv == bat == 0 and r['profile'] in ('flat', 'day', 'evening'):
        assert abs(litres-3672*3.25) < 1e-6
    data.setdefault(key, {})[r['variant']] = cost
    fuel_costs.setdefault(key, {})[r['variant']] = litres*p['fuel_rub_per_litre']
with (OUT/'holdout.csv').open(encoding='utf-8-sig', newline='') as f:
    held = list(csv.DictReader(f))
assert len(held) == 36
diagnostics = []
for s in json.loads((OUT/'summary.json').read_text(encoding='utf-8')):
    point = s['point']
    train = [costs for (place, year, profile), costs in data.items() if place == point and year <= 2017]
    variants = list(data[(point, 2017, 'flat')])
    assert len(variants) == s['variants']
    assert len(train) == 39 and all(set(x) == set(variants) for x in train)
    scores = {v: max(x[v]-min(x.values()) for x in train) for v in variants}
    robust = min(variants, key=lambda v: scores[v])
    baseline = min(variants, key=lambda v: data[(point, 2017, 'flat')][v])
    assert robust == s['robust_variant'] and baseline == s['baseline_variant']
    training_choices = {min(x, key=x.get) for x in train} | {robust, baseline}
    def on_upper_boundary(variant):
        solar, battery = variant.split('_bat')
        return int(solar[2:]) == s['pv_max'] or int(battery) == s['bat_max']
    assert any(on_upper_boundary(v) for v in training_choices) == s['upper_boundary_reached']
    assert not s['upper_boundary_reached']
    assert abs(scores[robust]-s['training_worst_regret_rub']) < 1e-6
    assert s['selection_status'] == ('no_clear_choice' if scores[robust] > p['user_max_regret_rub'] else 'within_assumed_threshold')
    for method, variant in [('robust', robust), ('baseline', baseline)]:
        values = []
        for r in [x for x in held if x['point'] == point]:
            key = (point, int(r['year']), r['profile'])
            regret = data[key][variant]-min(data[key].values())
            assert abs(regret-float(r[method+'_regret_rub'])) < 1e-6
            assert abs(min(data[key].values())-float(r['oracle_cost_rub'])) < 1e-6
            if method == 'baseline':
                assert abs(fuel_costs[key][variant]-float(r['baseline_fuel_rub'])) < 1e-6
                assert abs(regret/fuel_costs[key][variant]-float(r['baseline_regret_fraction_of_fuel'])) < 1e-9
            values.append(regret)
        assert abs(statistics.mean(values)-s[method+'_mean']) < 1e-6
        assert abs(max(values)-s[method+'_max']) < 1e-6
    for key, costs in data.items():
        if key[0] != point or key[2] == 'flat':
            continue
        regret = costs[baseline]-min(costs.values())
        diagnostics.append(dict(point=point, year=key[1], profile=key[2],
                                split=p['profiles'][key[2]]['set'], baseline=baseline,
                                regret_rub=regret, selected_fuel_rub=fuel_costs[key][baseline],
                                fraction_of_selected_fuel=regret/fuel_costs[key][baseline]))
with (OUT/'profile-errors.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(diagnostics[0]))
    writer.writeheader()
    writer.writerows(diagnostics)
result = dict(passed=True, seasonal_runs=len(rows), holdout_scenarios=len(held),
              profile_error_scenarios=len(diagnostics), verification_seconds=time.perf_counter()-begin,
              max_balance_error_kwh=max(float(r['max_balance_error_kwh']) for r in rows),
              checks=['unique rows', 'model and parameter hashes', 'costs and analytical diesel case',
                      'train and holdout separation', 'energy bounds', 'independent selections and losses',
                      'training optima and both selections below grid upper boundaries',
                      'price of wrong selection and fraction of selected fuel cost'])
(OUT/'independent-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
