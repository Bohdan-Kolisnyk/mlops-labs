"""Constants of the label definition (docs/design.md 5.1) and of the pipeline."""

ET = "America/New_York"

# Label: y = 1 if |r| > K * sigma, sigma = std of the same two-bar return at the same
# time slot over the previous SIGMA_DAYS trading days (design 5.1).
K = 1.5
SIGMA_DAYS = 20

# Bar c (the hourly bar that contains the post) must be followed by a full hourly bar c+1
# in the same session. The last full bar starts at 14:30, so c starts at 13:30 at the latest,
# i.e. the post time is in [09:30, 14:30) ET (design 5.2, "usable rows").
POST_SLOTS = ("09:30", "10:30", "11:30", "12:30", "13:30")

FEATURE_VERSION = "v1"

# Time-based split (design 5.4). The split is made on the post's session date.
TRAIN_END = "2025-06-30"
VAL_START = "2025-07-01"
VAL_END = "2025-12-31"
TEST_START = "2026-01-01"

# Quality gate thresholds (design 6.6). The first draft said 40% for empty texts; the monthly
# share measured on the archive is 7% to 45% (mean 24%), so 40% stopped two normal months.
EMPTY_TEXT_MAX_SHARE = 0.60
BARS_PER_FULL_DAY = 7
BARS_PER_HALF_DAY = 3
MIN_BATCH_ROWS = 10

# Files inside the DVC-tracked data/raw directory
POSTS_FILE = "train-00000-of-00001.parquet"
BARS_FILE = "gspc_1h.csv"
RAW_POST_COLUMNS = ["post_id", "datetime", "text", "has_media"]
