#!/usr/bin/env python3
"""Microbenchmark suite for Py2VM optimizer.

Measures:
  - CPython baseline (direct exec)
  - Py2VM baseline (opt_level=0, no tiering)
  - Py2VM opt_level=1 (peephole + CFG)
  - Py2VM opt_level=2 (+ superinstructions)
  - Py2VM opt_level=2 + tiering enabled
  - Py2VM opt_level=2 + all experimental (inline cache, guarded arith, range fast path)

Since inline cache, guarded arithmetic, and range fast path are always active
in the dispatch loop, we measure their effect indirectly through benchmarks
that specifically stress those paths (LOAD_GLOBAL-heavy, arithmetic-heavy,
range-loop-heavy).  The opt_level controls IR-level optimizations only.
"""

import time
import sys
import os

# ---------------------------------------------------------------------------
# Benchmark source snippets (executed as strings)
# ---------------------------------------------------------------------------

BENCH_ARITHMETIC_LOOP = """
def bench():
    s = 0
    for i in range(10000):
        s = s + i * 3 - 1
    return s
result = bench()
"""

BENCH_FLOAT_ARITHMETIC = """
def bench():
    s = 0.0
    for i in range(10000):
        s = s + float(i) * 1.5 - 0.25
    return s
result = bench()
"""

BENCH_LIST_APPEND = """
def bench():
    lst = []
    for i in range(10000):
        lst.append(i)
    return len(lst)
result = bench()
"""

BENCH_DICT_GET_SET = """
def bench():
    d = {}
    for i in range(5000):
        d[i] = i * 2
    s = 0
    for i in range(5000):
        s += d[i]
    return s
result = bench()
"""

BENCH_ATTRIBUTE_READ = """
class Obj:
    def __init__(self):
        self.x = 1
        self.y = 2
        self.z = 3

def bench():
    o = Obj()
    s = 0
    for i in range(10000):
        s += o.x + o.y + o.z
    return s
result = bench()
"""

BENCH_FUNCTION_CALL = """
def add(a, b):
    return a + b

def bench():
    s = 0
    for i in range(5000):
        s = add(s, i)
    return s
result = bench()
"""

BENCH_EXCEPTION_FREE_TRY = """
def bench():
    s = 0
    for i in range(5000):
        try:
            s += i
        except ValueError:
            pass
    return s
result = bench()
"""

BENCH_GENERATOR = """
def gen(n):
    for i in range(n):
        yield i

def bench():
    s = 0
    for v in gen(5000):
        s += v
    return s
result = bench()
"""

BENCH_GLOBAL_LOOKUP = """
CONSTANT_A = 42
CONSTANT_B = 17

def bench():
    s = 0
    for i in range(5000):
        s += CONSTANT_A + CONSTANT_B
    return s
result = bench()
"""

BENCH_NESTED_LOOP = """
def bench():
    s = 0
    for i in range(200):
        for j in range(200):
            s += i + j
    return s
result = bench()
"""

BENCHMARKS = [
    ("int_arithmetic",   BENCH_ARITHMETIC_LOOP),
    ("float_arithmetic", BENCH_FLOAT_ARITHMETIC),
    ("list_append",      BENCH_LIST_APPEND),
    ("dict_get_set",     BENCH_DICT_GET_SET),
    ("attribute_read",   BENCH_ATTRIBUTE_READ),
    ("function_call",    BENCH_FUNCTION_CALL),
    ("try_except_clean", BENCH_EXCEPTION_FREE_TRY),
    ("generator",        BENCH_GENERATOR),
    ("global_lookup",    BENCH_GLOBAL_LOOKUP),
    ("nested_loop",      BENCH_NESTED_LOOP),
]

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_cpython(source, n_iter=5):
    """Time direct CPython exec of the source, return best-of-n in seconds."""
    code_obj = compile(source, "<bench>", "exec")
    best = float("inf")
    for _ in range(n_iter):
        ns = {}
        t0 = time.perf_counter()
        exec(code_obj, ns)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def time_py2vm(source, opt_level, tiering, n_iter=5):
    """Time Py2VM execution of the source, return best-of-n in seconds."""
    import py2vm
    import optimizer as opt

    # Save state
    old_opt = py2vm._OPT_LEVEL
    old_tier = py2vm._TIERING_ENABLED

    py2vm.set_opt_level(opt_level)
    py2vm.set_tiering(tiering)
    # Clear caches so we get fresh optimization at the new level
    opt._DECODE_CACHE.clear() if hasattr(opt, '_DECODE_CACHE') else None
    opt._OPTIMIZE_CACHE.clear() if hasattr(opt, '_OPTIMIZE_CACHE') else None
    # Reset tier counters
    opt._TIER_COUNTERS.clear() if hasattr(opt, '_TIER_COUNTERS') else None

    code_obj = compile(source, "<bench>", "exec")
    best = float("inf")
    for _ in range(n_iter):
        # Clear caches between runs for consistent measurement
        opt._OPTIMIZE_CACHE.clear() if hasattr(opt, '_OPTIMIZE_CACHE') else None
        opt._TIER_COUNTERS.clear() if hasattr(opt, '_TIER_COUNTERS') else None
        t0 = time.perf_counter()
        py2vm.py2vm(code_obj)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)

    # Restore state
    py2vm.set_opt_level(old_opt)
    py2vm.set_tiering(old_tier)
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmarks():
    configs = [
        ("CPython",               None, None),
        ("Py2VM opt=0 (baseline)", 0,   False),
        ("Py2VM opt=1 (peephole)", 1,   False),
        ("Py2VM opt=2 (+supers)",  2,   False),
        ("Py2VM opt=2 +tiering",   2,   True),
    ]

    # Collect results: {bench_name: {config_label: time_secs}}
    results = {}

    print(f"Running {len(BENCHMARKS)} benchmarks x {len(configs)} configs (best-of-5)\n")

    for bench_name, source in BENCHMARKS:
        results[bench_name] = {}
        sys.stdout.write(f"  {bench_name:<20s}")
        sys.stdout.flush()
        for label, opt_level, tiering in configs:
            if opt_level is None:
                t = time_cpython(source)
            else:
                t = time_py2vm(source, opt_level, tiering)
            results[bench_name][label] = t
            sys.stdout.write(".")
            sys.stdout.flush()
        sys.stdout.write(" done\n")

    # Print results table
    col_labels = [c[0] for c in configs]
    col_w = max(len(l) for l in col_labels) + 2

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS (times in milliseconds, best-of-5)")
    print("=" * 100)

    header = f"{'Benchmark':<20s}"
    for label in col_labels:
        header += f"  {label:>{col_w}s}"
    header += f"  {'Slowdown':>10s}"
    print(header)
    print("-" * len(header))

    for bench_name, _ in BENCHMARKS:
        row = results[bench_name]
        cpython_t = row[col_labels[0]]
        line = f"{bench_name:<20s}"
        for label in col_labels:
            ms = row[label] * 1000
            line += f"  {ms:>{col_w}.2f}"
        # Slowdown: best Py2VM / CPython
        best_vm = min(row[label] for label in col_labels[1:])
        if cpython_t > 0:
            slowdown = best_vm / cpython_t
            line += f"  {slowdown:>9.1f}x"
        else:
            line += f"  {'N/A':>10s}"
        print(line)

    print("-" * len(header))

    # Summary: per-config geometric mean slowdown vs CPython
    print("\nSummary: Geometric mean slowdown vs CPython")
    print("-" * 60)
    import math
    for label in col_labels[1:]:
        ratios = []
        for bench_name, _ in BENCHMARKS:
            cpython_t = results[bench_name][col_labels[0]]
            vm_t = results[bench_name][label]
            if cpython_t > 0:
                ratios.append(vm_t / cpython_t)
        if ratios:
            geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            print(f"  {label:<35s} {geomean:>6.1f}x")

    # Optimization speedup relative to opt=0 baseline
    baseline_label = col_labels[1]  # "Py2VM opt=0 (baseline)"
    print(f"\nSpeedup vs {baseline_label}")
    print("-" * 60)
    for label in col_labels[2:]:
        speedups = []
        for bench_name, _ in BENCHMARKS:
            baseline_t = results[bench_name][baseline_label]
            opt_t = results[bench_name][label]
            if opt_t > 0:
                speedups.append(baseline_t / opt_t)
        if speedups:
            geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
            print(f"  {label:<35s} {geomean:>6.2f}x faster")

    return results


if __name__ == "__main__":
    run_benchmarks()
