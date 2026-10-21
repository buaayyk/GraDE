import argparse
import json
import numpy as np

def aggregate(args):
    with open(args.score_logps_path, "r") as f:
        score_logps = json.load(f)
    
    logps_array = np.array(score_logps['G_logp'])
    probs = np.mean(logps_array, axis=-1).tolist()
    
    with open(args.probs_path, "w") as f:
        json.dump(probs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_logps_path', type=str)
    parser.add_argument('--probs_path', type=str)
    args = parser.parse_args()
    aggregate(args)
