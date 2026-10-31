"""让 `pytest` 从仓库根导入 `src.*`（tests/ 不在包内，需要显式加 sys.path）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
