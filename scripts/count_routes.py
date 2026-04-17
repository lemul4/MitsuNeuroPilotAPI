#!/usr/bin/env python3
"""Count route files in data/data_routes excluding specified towns.

Usage:
  python scripts/count_routes.py [--root data/data_routes] [--exclude Town10,Town11]

Outputs total .xml routes and counts grouped by top-level set and by route-type (parent dir).
"""
import os
import argparse
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description="Count route XML files in data/data_routes")
    p.add_argument('--root', default=os.path.join('data', 'data_routes'), help='root folder to scan')
    p.add_argument('--ext', default='.xml', help='route file extension to count')
    p.add_argument('--exclude', default='Town11,Town12,Town13,Town14,Town15',
                   help='comma-separated town names to exclude (case-insensitive)')
    return p.parse_args()


def main():
    args = parse_args()
    excludes = {x.strip().lower() for x in args.exclude.split(',') if x.strip()}

    total = 0
    by_top = Counter()
    by_type = Counter()

    root = args.root
    if not os.path.isdir(root):
        print(f'Root folder not found: {root}')
        return 2

    for dirpath, dirs, files in os.walk(root):
        # skip any directory that contains an excluded town name in its path
        parts = [p.lower() for p in os.path.relpath(dirpath, root).split(os.sep) if p and p != os.curdir]
        if any(ex in part for part in parts for ex in excludes):
            continue

        for fn in files:
            if not fn.lower().endswith(args.ext.lower()):
                continue
            full = os.path.join(dirpath, fn)
            # check full path for excluded towns as well (covers filenames containing town)
            rel = os.path.relpath(full, root)
            if any(ex in p for p in rel.lower().split(os.sep) for ex in excludes):
                continue

            total += 1

            rel_parts = rel.split(os.sep)
            top = rel_parts[0] if len(rel_parts) >= 2 else rel_parts[0]
            by_top[top] += 1

            parent = os.path.basename(os.path.dirname(full))
            by_type[parent] += 1

    print(f'Total {args.ext} routes (excluding {", ".join(sorted(excludes))}): {total}')

    if by_top:
        print('\nCounts by top-level set:')
        for k, v in by_top.most_common():
            print(f'  {k}: {v}')

    if by_type:
        print('\nCounts by route-type (parent directory of file):')
        for k, v in by_type.most_common():
            print(f'  {k}: {v}')


if __name__ == '__main__':
    raise SystemExit(main())
