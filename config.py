
import os
from collections import OrderedDict


BACKEND = os.environ.get("TOFFEE_BACKEND", "openrouter")

MODELS = OrderedDict([
    ("cost_effective", "deepseek/deepseek-v3.2"),
    ("capable",        "anthropic/claude-sonnet-4.6"),
    ("premium",        "openai/gpt-5.4"),
])


_COST_OVERRIDE = os.environ.get("TOFFEE_COST_MODEL", "")
if _COST_OVERRIDE:
    MODELS = OrderedDict(MODELS)
    MODELS["cost_effective"] = _COST_OVERRIDE
MODEL_TIERS = {v: k for k, v in MODELS.items()}
MODEL_LIST = list(MODELS.values())
N_MODELS = len(MODEL_LIST)

MODEL_COSTS_PER_M = {
    "deepseek/deepseek-v3.2":      (0.30,  0.88),
    "anthropic/claude-sonnet-4.6": (3.0,  15.0),
    "openai/gpt-5.4":              (2.5,  10.0),
    "anthropic/claude-opus-4.6":   (15.0, 75.0),
}


JUDGE_MODEL = "google/gemini-3.1-pro-preview"


TASK_REALIZE_MODEL = "anthropic/claude-opus-4.6"


EFFORT_LEVELS = {"standard": 2048, "moderate": 4096, "extended": 8192}
EFFORT_LIST = list(EFFORT_LEVELS.keys())
N_EFFORTS = len(EFFORT_LIST)


HISTORY_MODES = {"short": 6000, "mid": 16000, "long": 32000}
HISTORY_LIST = list(HISTORY_MODES.keys())
N_HISTORIES = len(HISTORY_LIST)


MCTS_MAX_ITERATIONS = 25

C_PUCT = float(os.environ.get("TOFFEE_CPUCT", "1.0"))


LEVEL_BUDGET_SCALING = os.environ.get("TOFFEE_LEVEL_BUDGET", "1") == "1"


K_MAX = 3

REWARD_LAMBDA = float(os.environ.get("TOFFEE_LAMBDA", "0.25"))


ALPHA_B = 1.5
RIDGE_LAMBDA = 1.0


LCM_ENABLED = os.environ.get("TOFFEE_LCM", "1") == "1"


ROUTER_MODE = os.environ.get("TOFFEE_ROUTER", "lcm")


D_STATE = 11
D_ACTION = 17


UNIFIED_TOOL_NAMES = (
    "list_tables", "get_table_schema", "execute_sql", "execute_python",
    "read_file", "list_directory", "write_file", "run_bash",
)
N_TOOLS = len(UNIFIED_TOOL_NAMES)


MIN_ACCEPT_QUALITY = 0.45


# Computed condition: an unfiltered scan (no filter, join, or aggregation)
# whose output exceeds this many rows cannot back a stated value.
WIDE_SCAN_ROW_LIMIT = 100


# Metered tool execution, charged against B_q and the step reward alongside
# billed LLM tokens. Default: AWS on-demand vCPU rate (USD per vCPU-hour).
# The execution-priced search arm overrides this with a cloud-warehouse rate.
EXEC_VCPU_RATE_PER_HOUR = float(os.environ.get("TOFFEE_EXEC_RATE", "0.05"))


# The LCM reads the execution state alone by default; the provenance-derived
# state dimensions (level one-hot, source span, format span) stay zeroed unless the
# "+ provenance features" ablation arm turns them on.
PROVENANCE_FEATURES = os.environ.get("TOFFEE_PROVENANCE_FEATURES", "0") == "1"


# Admission ends with an independent replay of the whole check battery.
ADMISSION_REPLAY = os.environ.get("TOFFEE_ADMISSION_REPLAY", "1") == "1"


ACCEPT_RULE = os.environ.get("TOFFEE_ACCEPT_RULE", "fact")


ACCEPT_CUTS = {"all": 0.0, "partial": 0.5, "fact": 0.9, "exact": 0.99}


VH_MODE = os.environ.get("TOFFEE_VH_MODE", "mean")


TOOL_TIMEOUT_S = 30
MAX_CONSECUTIVE_ERRORS = 5
MAX_STEP_COUNT = 20


SOURCE_FORMATS = ("sqlite_table", "csv_table", "excel_sheet", "md_struct")
N_FORMATS = len(SOURCE_FORMATS)


B_SKEL_BASE = 3
TAU_ND = 0.10


MAX_UNITS_PER_ENV = 40


JOIN_VERIFY_BUDGET = 20
CHAIN_VERIFY_BUDGET = 20
BRIDGE_MAX = 8
VALUE_SAMPLE_N = 200


ROLE_GATE = os.environ.get("TOFFEE_ROLE_GATE", "1") == "1"


JOIN_FANOUT_MAX = int(os.environ.get("TOFFEE_JOIN_FANOUT_MAX", "64"))


PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_SOURCE_CATALOG = os.environ.get(
    "TOFFEE_CATALOG", os.path.join(PACKAGE_ROOT, "data", "data_source_catalog.json"))
