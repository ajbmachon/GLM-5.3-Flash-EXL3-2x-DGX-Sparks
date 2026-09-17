"""Generate an explicit TP3 overlay beside an already measured native bundle.

No downloads, lifecycle commands, model changes or default configuration edits.
"""
import argparse
import hashlib
from pathlib import Path

from manifest import verify_artifacts

STOCK_SHA = "fe07cf3cd1928d0a189e793579a7d2dd529f75617a55f620ee14a0a9d3b20121"


def prepare(stock, bundle):
    stock, bundle = Path(stock), Path(bundle)
    source = stock.read_bytes()
    if hashlib.sha256(source).hexdigest() != STOCK_SHA:
        raise ValueError("stock EXL3 source changed; review compatibility before repinning")
    verify_artifacts(bundle)
    footer = '''
# Explicit TP3 ABI2 cooperative adapter; complete manifest required on all ranks.
import runpy as _coop_runpy
import sys as _coop_sys
_coop_setup = _coop_runpy.run_path("/root/.cache/vllm/cooperative_moe/runtime.py")
_coop_setup["install"](_coop_sys.modules[__name__], library_root="/root/.cache/vllm/cooperative_moe", enabled=True)
'''
    output = bundle / "exl3-tp3.py"
    with output.open("xb") as handle:
        handle.write(source + footer.encode())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.stock, args.bundle))
