"""Seed Finder - finds dramatic demo seeds"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import EvalEngine

def score_episode(engine, seed):
    state = engine.reset(seed=seed)
    max_infected = 0; max_damage = 0; total_steps = 0
    had_block = False; had_quarantine = False
    while not engine.done:
        state = engine.step()
        if state is None: break
        total_steps = state['timestep']
        inf = state['counts']['infected_total']
        if inf > max_infected: max_infected = inf
        dmg = state['server_damage']
        if dmg > max_damage: max_damage = dmg
        for e in state.get('events', []):
            if e['type'] == 'block': had_block = True
            if e['type'] == 'quarantine': had_quarantine = True
    outcome = state['outcome']
    score = 0.0
    if outcome == 'win':
        score += 100.0 + max_infected * 15.0 + max_damage * 0.5
        score += (10.0 if had_block else 0) + (20.0 if had_quarantine else 0)
        if 30 < total_steps < 150: score += 30.0
    elif outcome == 'survived':
        score += 20.0 + max_infected * 5.0
    else:
        score += 10.0 + max_infected * 3.0
    return {'seed': seed, 'outcome': outcome, 'score': score,
            'max_infected': max_infected, 'max_damage': round(max_damage, 1),
            'steps': total_steps, 'had_block': had_block, 'had_quarantine': had_quarantine}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    engine = EvalEngine(checkpoint_dir=args.checkpoint, device='cpu')
    engine.deterministic = True
    results = []
    for seed in range(args.start, args.start + args.count):
        r = score_episode(engine, seed)
        results.append(r)
        if (seed - args.start + 1) % 20 == 0:
            print(f"  Tested {seed - args.start + 1}/{args.count} seeds...")
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"\n{'='*70}\n  TOP {args.top} DEMO SEEDS\n{'='*70}")
    print(f"{'Rank':>4} {'Seed':>6} {'Outcome':>10} {'Score':>7} {'MaxInf':>7} {'MaxDmg':>7} {'Steps':>6}")
    for i, r in enumerate(results[:args.top]):
        print(f"{i+1:>4} {r['seed']:>6} {r['outcome']:>10} {r['score']:>7.1f} "
              f"{r['max_infected']:>7} {r['max_damage']:>7.1f} {r['steps']:>6}")
    print(f"\nBest seed: {results[0]['seed']}")

if __name__ == '__main__':
    main()
