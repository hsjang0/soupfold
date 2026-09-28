"""OpenDDE: same code path and flags as scripts/run_protenix.py, plus --checkpoint.

  python scripts/run_opendde.py --mode native --input examples/9mnb/9mnb.json --checkpoint <opendde.pt> <data flags>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_protenix import main  # noqa: E402

if __name__ == "__main__":
    main("opendde")
