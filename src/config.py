"""Locked project constants for the DS1 teen mental-health prediction track.

Dataset: https://www.kaggle.com/datasets/argonnxx/teen-mental-health
Decisions: see internal_docs/master_plan_source.md
"""

SEED = 42
TEST_SIZE = 0.2

TARGET = "digital_wellbeing_flag"
TASK = "multiclass"
TARGET_CLASSES = ("Healthy", "Moderate", "At Risk")

# Expected class counts from the locked Kaggle snapshot (soft-check only).
EXPECTED_TARGET_COUNTS = {
    "Healthy": 306,
    "Moderate": 743,
    "At Risk": 151,
}

LIFESTYLE_FEATURES = [
    "age",
    "gender",
    "daily_social_media_hours",
    "platform_usage",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "social_interaction_level",
]

# Documented exclusions — not used as predictors (lifestyle-only policy).
# Psycho scales overlap with mental_health_risk_score (= stress + anxiety + addiction).
EXCLUDED_COLUMNS = [
    "stress_level",
    "anxiety_level",
    "addiction_level",
    "mental_health_risk_score",
    "depression_label",
    "sleep_quality",
]

DATA_FILENAME = "Teen_Mental_Health.csv"
DATASET_URL = "https://www.kaggle.com/datasets/argonnxx/teen-mental-health"
ORIGIN_ARCHIVE_PATH = "internal_docs/data/origin_database/Teen_Mental_Health.csv"

EXPECTED_ROWS = 1200
EXPECTED_COLS = 16

ALL_COLUMNS = [
    "age",
    "gender",
    "daily_social_media_hours",
    "platform_usage",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "social_interaction_level",
    "stress_level",
    "anxiety_level",
    "addiction_level",
    "depression_label",
    "mental_health_risk_score",
    "sleep_quality",
    "digital_wellbeing_flag",
]
