"""Expanded-grid diagnostic on saved inputs; preserves experiment 1."""
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiment2'
spec = importlib.util.spec_from_file_location('vlad_balance', ROOT/'calculate.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    begin = time.perf_counter()
    p = json.loads((OUT/'parameters.json').read_text(encoding='utf-8'))
    assert p['grid_limits'] == [[60, 120], [120, 240]]
    model.checks(p)
    manifest = json.loads((model.DATA/'weather-manifest.json').read_text(encoding='utf-8'))
    records, summaries, errors = [], [], []
    for source in manifest:
        raw = (model.DATA/source['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source['sha256']
        years = {}
        for r in json.loads(raw)['outputs']['hourly']:
            local = dt.datetime.strptime(r['time'], '%Y%m%d:%H%M') + dt.timedelta(hours=source['utc_offset'])
            if local.month in p['months']:
                years.setdefault(local.year, []).append((local.hour, r['P']/1000))
        scenarios = {}
        for year, weather in years.items():
            for name, profile in p['profiles'].items():
                split = 'train' if year <= 2017 else 'holdout'
                if profile['set'] == split:
                    scenarios[(year, name)] = (weather, [model.load_at(profile, h) for h, _ in weather])
        costs, fuels = {k: {} for k in scenarios}, {k: {} for k in scenarios}
        for pv_max, bat_max in p['grid_limits']:
            variants = [(0, 0)] + [(pv, bat) for pv in range(10, pv_max+1, 10) for bat in range(0, bat_max+1, 20)]
            for key, (weather, loads) in scenarios.items():
                for pv, bat in variants:
                    variant = f'pv{pv}_bat{bat}'
                    if variant in costs[key]:
                        continue
                    result = model.simulate(weather, loads, pv, bat, p)
                    assert result['litres'] is not None
                    total = result['litres']*p['fuel_rub_per_litre'] + pv*p['pv_season_rub_per_kw'] + bat*p['battery_season_rub_per_kwh']
                    costs[key][variant], fuels[key][variant] = total, result['litres']*p['fuel_rub_per_litre']
                    records.append(dict(point=source['point'], year=key[0], profile=key[1], variant=variant,
                                        pv_kw=pv, usable_battery_kwh=bat, seasonal_cost_rub=total, **result))
            train = [k for k in costs if k[0] <= 2017]
            held = [k for k in costs if k[0] >= 2018]
            robust, baseline, worst = model.choose(costs, train, (2017, 'flat'))
            lookup = {f'pv{pv}_bat{bat}': (pv, bat) for pv, bat in variants}
            training_choices = {min(costs[k], key=costs[k].get) for k in train}
            edge = any(lookup[v][0] == pv_max or lookup[v][1] == bat_max for v in training_choices | {robust, baseline})
            if not edge:
                break
        for key in held:
            oracle = min(costs[key].values())
            errors.append(dict(point=source['point'], year=key[0], profile=key[1],
                               robust=robust, baseline=baseline, oracle_cost_rub=oracle,
                               robust_regret_rub=costs[key][robust]-oracle,
                               baseline_regret_rub=costs[key][baseline]-oracle,
                               baseline_fuel_rub=fuels[key][baseline],
                               baseline_regret_fraction_of_fuel=(costs[key][baseline]-oracle)/fuels[key][baseline]))
        local = [x for x in errors if x['point'] == source['point']]
        summaries.append(dict(point=source['point'], pv_max=pv_max, bat_max=bat_max,
                              variants=len(variants), upper_boundary_reached=edge,
                              robust_variant=robust, baseline_variant=baseline,
                              training_worst_regret_rub=worst[robust],
                              selection_status='no_clear_choice' if worst[robust] > p['user_max_regret_rub'] else 'within_assumed_threshold',
                              robust_mean=statistics.mean(x['robust_regret_rub'] for x in local),
                              robust_max=max(x['robust_regret_rub'] for x in local),
                              baseline_mean=statistics.mean(x['baseline_regret_rub'] for x in local),
                              baseline_max=max(x['baseline_regret_rub'] for x in local)))
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    for name, rows in [('seasonal-results.csv', records), ('holdout.csv', errors)]:
        with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    dump('summary.json', summaries)
    dump('run.json', dict(seasonal_runs=len(records), calculation_seconds=time.perf_counter()-begin,
                          code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          model_sha256=hashlib.sha256((ROOT/'calculate.py').read_bytes()).hexdigest(),
                          parameters_sha256=hashlib.sha256((OUT/'parameters.json').read_bytes()).hexdigest(),
                          source_sha256=[x['sha256'] for x in manifest]))


if __name__ == '__main__':
    main()
