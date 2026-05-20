#!/usr/bin/env python3
import argparse
import json


def main():
    p = argparse.ArgumentParser("Compare robust run vs clean baseline")
    p.add_argument("--baseline", required=True, help="Path to baseline final_test_metrics.json")
    p.add_argument("--robust", required=True, help="Path to robust final_test_metrics.json")
    args = p.parse_args()

    with open(args.baseline, "r", encoding="utf-8") as f:
        b = json.load(f)
    with open(args.robust, "r", encoding="utf-8") as f:
        r = json.load(f)

    keys = ["test_clean_clean_loss", "test_blur_blur_loss", "test_clean_blur_loss"]
    print("metric,baseline,robust,delta(robust-baseline)")
    for k in keys:
        bv = float(b[k])
        rv = float(r[k])
        print(f"{k},{bv:.6f},{rv:.6f},{(rv - bv):+.6f}")


if __name__ == "__main__":
    main()
