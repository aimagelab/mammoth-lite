import sys
import os

mammoth_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mammoth_path)

os.environ["MAMMOTH_BASE_PATH"] = mammoth_path