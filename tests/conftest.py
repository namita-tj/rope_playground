import os
import sys

# Let tests import the modules in the repository root (positional_encodings, attention, experiment).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
