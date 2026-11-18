"""Offline, standard-library research calculation; parameters fixed before run.

P from PVGIS is W for a 1 kWp array at hourly timestamps. We use P/1000
as the mean kW over the corresponding one-hour model interval. Fixed local
standard time, not historical civil time; no second PV inverter loss.
"""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'issledovanie-data'


def save_json(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(name, rows):
    with (ROOT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_at(profile, hour):
    return profile['high_kw'] if (hour-profile['start']) % 24 < profile['hours'] else profile['low_kw']


def fuel(power, curve):
    if power == 0:
        return 0.0
    for (x, y), (xx, yy) in zip(curve, curve[1:]):
        if x <= power <= xx:
            return y + (yy-y)*(power-x)/(xx-x)
    return None  # Unknown consumption is never silently zeroed.


def step(pv, load, soc, capacity, p):
    assert min(pv, load, soc, capacity) >= 0
    assert soc <= capacity + 1e-9
    eta_c, eta_d = p['charge_efficiency'], p['discharge_efficiency']
    limit = capacity * p['battery_power_per_kwh']
    used_pv = min(pv, load)
    deficit = load-used_pv
    charge = discharge = gen = 0.0
    if deficit <= 1e-12:
        charge = min(pv-used_pv, limit, max(0, capacity-soc)/eta_c)
    elif deficit <= limit+1e-12 and deficit/eta_d <= soc+1e-12:
        discharge = deficit
    else:
        gen = max(p['generator_min_kw'], deficit)
        if gen > p['generator_kw']:
            raise ValueError('Generator cannot meet load')
        charge = min(gen-deficit, limit, max(0, capacity-soc)/eta_c)
    next_soc = soc + charge*eta_c - discharge/eta_d
    spill = pv + gen + discharge - load - charge
    loss = charge*(1-eta_c) + discharge*(1/eta_d-1)
    residual = abs(pv+gen-load-spill-loss-(next_soc-soc))
    assert residual < 1e-8
    assert -1e-9 <= next_soc <= capacity+1e-9
    assert spill >= -1e-9 and charge*discharge == 0
    assert charge <= limit+1e-9 and discharge <= limit+1e-9
    assert gen == 0 or p['generator_min_kw'] <= gen <= p['generator_kw']
    return max(0, min(capacity, next_soc)), gen, fuel(gen, p['fuel_curve_lph']), residual


def simulate(rows, loads, pv_kw, capacity, p):
    soc = p['initial_usable_energy_kwh']
    litres = 0.0
    unknown = on_hours = 0
    residual = 0.0
    for (hour, solar), load in zip(rows, loads):
        soc, gen, amount, err = step(pv_kw*solar, load, soc, capacity, p)
        on_hours += gen > 0
        unknown += amount is None
        litres += amount if amount is not None else 0
        residual = max(residual, err)
    return dict(litres=litres if not unknown else None, known_litres=litres,
                unknown_fuel_hours=unknown, generator_hours=on_hours,
                initial_soc_kwh=p['initial_usable_energy_kwh'], final_soc_kwh=soc,
                max_balance_error_kwh=residual)


def choose(costs, train_keys, baseline_key):
    if any(costs[k][v] is None for k in train_keys for v in costs[k]):
        return None  # No cost ranking with unknown absolute totals.
    variants = list(costs[baseline_key])
    worst = {v: max(costs[k][v]-min(costs[k].values()) for k in train_keys) for v in variants}
    return min(variants, key=lambda v: (worst[v], variants.index(v))), min(variants, key=lambda v: (costs[baseline_key][v], variants.index(v))), worst


def checks(p):
    tests = []
    def check(name, ok):
        assert ok, name
        tests.append(dict(check=name, passed=True))
    rows = [(h, 0.0) for h in range(24)]
    a = simulate(rows, [9.0]*24, 0, 0, p)
    check('No solar, no battery: analytical 24*(1+0.25*9) = 78 L', abs(a['litres']-78) < 1e-9)
    check('Zero load, empty storage: zero fuel', simulate(rows, [0.0]*24, 0, 0, p)['litres'] == 0)
    soc, gen, litres, _ = step(10, 2, 0, 20, p)
    check('PV surplus charges with efficiency', abs(soc-7.6) < 1e-9 and gen == 0 and litres == 0)
    soc, gen, _, _ = step(0, 2, 0, 20, p)
    check('Minimum diesel output charges surplus', gen == 5 and abs(soc-2.85) < 1e-9)
    soc, gen, _, _ = step(0, 2, 10, 20, p)
    check('Sufficient battery turns diesel off', gen == 0 and abs(soc-(10-2/0.95)) < 1e-9)
    _, gen, _, _ = step(0, 12, 20, 20, p)
    check('Battery power limit forces diesel on', gen == 12)
    partial = dict(p, fuel_curve_lph=[[10, 3.5], [20, 6.0]])
    a = simulate(rows, [9.0]*24, 0, 0, partial)
    check('Missing fuel curve produces unknown totals', a['litres'] is None and a['unknown_fuel_hours'] == 24)
    check('Unknown costs refuse ranking', choose({(2005, 'flat'): {'a': None}}, [(2005, 'flat')], (2005, 'flat')) is None)
    try:
        step(0, 21, 0, 0, p)
        rejected = False
    except ValueError:
        rejected = True
    check('Load above generator rating rejected', rejected)
    check('All profiles have equal daily energy 216 kWh', all(sum(load_at(x,h) for h in range(24)) == 216 for x in p['profiles'].values()))
    check('Training and held-out profile names disjoint', not ({k for k,x in p['profiles'].items() if x['set']=='train'} & {k for k,x in p['profiles'].items() if x['set']=='holdout'}))
    return tests


def main():
    started = dt.datetime.now(dt.timezone.utc)
    begin = time.perf_counter()
    p = json.loads((ROOT/'parameters.json').read_text(encoding='utf-8'))
    tests = checks(p)
    manifest = json.loads((DATA/'weather-manifest.json').read_text(encoding='utf-8'))
    provenance = []
    records = []
    comparisons = []
    summaries = []
    variants = {f'pv{pv}_bat{bat}': (pv, bat) for pv,bat in p['variants']}
    for source in manifest:
        path = DATA / source['file']
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == source['sha256']
        data = json.loads(raw)
        assert data['inputs']['pv_module']['peak_power'] == 1
        assert data['inputs']['pv_module']['system_loss'] == 14
        series = data['outputs']['hourly']
        assert len(series) == source['rows'] == 166536
        years = {}
        previous = None
        for r in series:
            t = dt.datetime.strptime(r['time'], '%Y%m%d:%H%M')
            assert previous is None or (t-previous).total_seconds() == 3600
            previous = t
            assert math.isfinite(r['P']) and r['P'] >= 0
            local = t + dt.timedelta(hours=source['utc_offset'])
            if local.month in p['months']:
                years.setdefault(local.year, []).append((local.hour, r['P']/1000))
        assert sorted(years) == list(range(2005,2024))
        assert all(len(x)==153*24 for x in years.values())
        provenance.append(dict(point=source['point'], file='../issledovanie-data/'+path.name,
                               sha256=digest, url=source['url'], rows=len(series), season_hours=3672,
                               utc_offset=source['utc_offset']))
        costs = {}
        by_scenario = {}
        for year, rows in years.items():
            for profile, spec in p['profiles'].items():
                # Only training profiles in early years; only unseen profiles in holdout years.
                split = 'train' if year <= p['train_years'][1] else 'holdout'
                if spec['set'] != split:
                    continue
                key = (year,profile)
                costs[key] = {}
                by_scenario[key] = {}
                loads = [load_at(spec, h) for h,solar in rows]
                for variant,(pv,bat) in variants.items():
                    result = simulate(rows, loads, pv, bat, p)
                    cost = None if result['litres'] is None else result['litres']*p['fuel_rub_per_litre'] + pv*p['pv_season_rub_per_kw'] + bat*p['battery_season_rub_per_kwh']
                    costs[key][variant] = cost
                    by_scenario[key][variant] = result
                    records.append(dict(point=source['point'], year=year, profile=profile, split=split, variant=variant,
                                        pv_kw=pv, usable_battery_kwh=bat, seasonal_cost_rub=cost, **result))
        train = [k for k in costs if k[0] <= p['train_years'][1]]
        holdout = [k for k in costs if k[0] >= p['holdout_years'][0]]
        robust, baseline, worst = choose(costs, train, (p['baseline_year'],'flat'))
        assert len(train)==39 and len(holdout)==12
        for key in holdout:
            best = min(costs[key].values())
            comparisons.append(dict(point=source['point'],year=key[0],profile=key[1],robust=robust,baseline=baseline,
                                    oracle_cost_rub=best,robust_regret_rub=costs[key][robust]-best,
                                    baseline_regret_rub=costs[key][baseline]-best))
        local = [x for x in comparisons if x['point']==source['point']]
        summaries.append(dict(point=source['point'], train_scenarios=len(train), holdout_scenarios=len(holdout),
                              robust_variant=robust, baseline_variant=baseline,
                              training_worst_regret_rub=worst[robust],
                              selection_status='no_clear_choice' if worst[robust]>p['user_max_regret_rub'] else 'within_assumed_threshold',
                              holdout_robust_mean_regret_rub=statistics.mean(x['robust_regret_rub'] for x in local),
                              holdout_baseline_mean_regret_rub=statistics.mean(x['baseline_regret_rub'] for x in local),
                              holdout_robust_max_regret_rub=max(x['robust_regret_rub'] for x in local),
                              holdout_baseline_max_regret_rub=max(x['baseline_regret_rub'] for x in local),
                              robust_better_scenarios=sum(x['robust_regret_rub']<x['baseline_regret_rub']-1e-7 for x in local),
                              robust_worse_scenarios=sum(x['robust_regret_rub']>x['baseline_regret_rub']+1e-7 for x in local)))
    tests.append(dict(check='All 1530 seasonal runs: energy balance, SOC, power and generator bounds checked every hour; no unknown fuel totals', passed=all(r['litres'] is not None and r['max_balance_error_kwh']<1e-8 for r in records)))
    assert tests[-1]['passed'] and len(records)==1530
    write_csv('seasonal-results.csv', records)
    write_csv('holdout.csv', comparisons)
    save_json('source-manifest.json', provenance)
    save_json('summary.json', summaries)
    save_json('verification.json', tests)
    save_json('run.json', dict(started_utc=started.isoformat(),finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                              calculation_seconds=time.perf_counter()-begin,seasonal_runs=len(records),hourly_steps=len(records)*3672,
                              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              parameters_sha256=hashlib.sha256((ROOT/'parameters.json').read_bytes()).hexdigest(),
                              tokens_measured=False))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
    print('PASS:',len(tests),'checks;', len(records),'seasonal runs')


if __name__ == '__main__':
    main()
