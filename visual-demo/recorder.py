"""Episode Recorder - records to JSON for offline replay"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import EvalEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--stochastic', action='store_true', default=False)
    args = parser.parse_args()
    engine = EvalEngine(checkpoint_dir=args.checkpoint, device='cpu')
    engine.deterministic = not args.stochastic
    print(f"Recording episode with seed {args.seed}...")
    frames = []
    state = engine.reset(seed=args.seed)
    frames.append(state)
    while not engine.done:
        state = engine.step()
        if state is None: break
        frames.append(state)
    if args.output is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f'seed_{args.seed}.json')
    with open(args.output, 'w') as f:
        json.dump(frames, f)
    print(f"Recorded {len(frames)} frames -> {args.output}")
    print(f"Outcome: {frames[-1].get('outcome', 'unknown')}")

if __name__ == '__main__':
    main()
