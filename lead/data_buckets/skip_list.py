import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone

CACHE_BUILD_FAILURES_FILENAME = "cache_build_failures.jsonl"


def sample_key(route_path: str, seq: int) -> str:
    normalized = os.path.normpath(route_path)
    route = os.path.basename(normalized)
    scenario = os.path.basename(os.path.dirname(normalized))
    return f"{scenario}/{route}/{int(seq):04d}"


def cache_build_failures_path(bucket_collection_path: str) -> str:
    return os.path.join(bucket_collection_path, CACHE_BUILD_FAILURES_FILENAME)


def load_cache_build_failure_keys(bucket_collection_path: str) -> set[str]:
    path = cache_build_failures_path(bucket_collection_path)
    if not os.path.exists(path):
        return set()

    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("key")
            if key:
                keys.add(key)
    return keys


def append_cache_build_failure(
    bucket_collection_path: str,
    route_dir: str,
    seq: int,
    error_type: str,
    error: str,
    index: int | None = None,
) -> str:
    key = sample_key(route_dir, seq)
    record = {
        "key": key,
        "route_dir": route_dir,
        "seq": int(seq),
        "error_type": error_type,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if index is not None:
        record["index"] = int(index)

    path = cache_build_failures_path(bucket_collection_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
    return key


def skipped_count_for_route(keys: Iterable[str], route_path: str) -> int:
    normalized = os.path.normpath(route_path)
    route = os.path.basename(normalized)
    scenario = os.path.basename(os.path.dirname(normalized))
    prefix = f"{scenario}/{route}/"
    return sum(1 for key in keys if key.startswith(prefix))
