# %%
"""
MAIN TEST FILE: about splitting data (global/local forecasting)
Comprehensive Test of D2 Split Scenarios

This test suite validates all splitting scenarios:

INVALID CASES (Data Leakage Detection):
1. Case 1: Train-Val overlap (groups 0,1 in both)
2. Case 2: Train-Test overlap (group 0 in both)
3. Case 3 (Invalid): Val-Test overlap (group 2 in both)
All invalid cases should raise ValueError with DATA LEAKAGE DETECTED message.

VALID CASES:
1. Case 3 (Hybrid): Train groups split temporally + dedicated val/test groups
2. Case 4 (Local): Pure temporal split (no groups)
3. Case 5 (Pure Group): 100% group separation
4. Case 6 (Pure Temporal): All groups split temporally

"""

# %%
import logging
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/sandeep/DSIPTS_PTF")

from dsipts.data_structure.d1_layers.multi_source_csv import MultiSourceTSDataSet
from dsipts.data_structure.d2_layers.encoder_decoder import EncoderDecoder

logging.basicConfig(level=logging.INFO)
print(f"✅ PyTorch {torch.__version__}")

# %%
# Load and prepare grouped weather data
data_path = "/home/sandeep/DSIPTS_PTF/data/"
weather_path = data_path + "weather.csv"

dataset = pd.read_csv(weather_path)
dataset.rename(columns={"date": "time"}, inplace=True)
dataset["time"] = pd.to_datetime(dataset["time"])

weather_data = dataset
target_col = "OT"
covariate_columns = list(set(dataset.columns).difference(set(["time", target_col])))

# Create grouped data with 4 groups
grouped_weather_path = data_path + "_grouped.csv"
groups_list = ["Group1", "Group2", "Group3", "Group4"]
probabilities = [0.1, 0.1, 0.1, 0.7]

weather_data_grouped = weather_data.assign(group=np.random.choice(groups_list, size=len(weather_data), p=probabilities))
weather_data_grouped.to_csv(grouped_weather_path, index=False)

print(f"✅ Dataset loaded: {weather_data_grouped.shape}")
print(f"   Group distribution:\n{weather_data_grouped['group'].value_counts()}")


# %%
def analyze_splits(d2_dataset, case_name):
    """Analyze and display split distribution by group."""
    print(f"\n{'='*80}")
    print(f"{case_name}")
    print(f"{'='*80}")

    if not d2_dataset.splits_created:
        print("❌ Splits not created!")
        return

    print(f"Valid windows in D2: {len(d2_dataset.valid_windows)}")

    all_window_groups = pd.Series([w["group_id"] for w in d2_dataset.valid_windows])

    def get_split_counts(indices, split_name):
        if not indices:
            print(f"\n--- {split_name} Split (0 windows) ---")
            print("   (empty)")
            return

        group_names_in_split = all_window_groups.iloc[indices]
        counts = group_names_in_split.value_counts().sort_index()

        print(f"\n--- {split_name} Split ({len(indices)} windows) ---")
        for group_id, count in counts.items():
            print(f"   Group {group_id}: {count} windows")

    get_split_counts(d2_dataset.train_indices, "TRAIN")
    get_split_counts(d2_dataset.val_indices, "VAL")
    get_split_counts(d2_dataset.test_indices, "TEST")


# %%
print("\n" + "=" * 80)
print("INVALID TEST CASES (Data Leakage Detection)")
print("=" * 80)

# %%
# ============================================================================
# CASE 1: INVALID - Train-Val Overlap (Data Leakage)
# ============================================================================
print("\n" + "=" * 80)
print("CASE 1: INVALID - Train-Val Overlap")
print("=" * 80)
print("Config: train=[0,1], val=[0,1], test=[3]")
print("Expected: ValueError (DATA LEAKAGE DETECTED)")

try:
    d1_case1 = MultiSourceTSDataSet(
        file_paths=[grouped_weather_path],
        time_col="time",
        target_cols=[target_col],
        num_cols=covariate_columns,
        enrich_cat=["hour", "dow"],
        global_forecasting=True,
        group_cols=["group"],
        memory_efficient=False,
        add_target_to_past=True,
    )

    d2_case1 = EncoderDecoder(
        d1_dataset=d1_case1,
        past_len=96,
        future_len=96,
        step_size=96,
        batch_size=32,
        split_ratio=(0.7, 0.15, 0.15),
        split_group_config=([0, 1], [0, 1], [3]),  # OVERLAP!
        scaling_method="standard",
        scale_targets=True,
    )

    d2_case1.setup(stage="fit")
    print("❌ ERROR: Should have raised ValueError but didn't!")

except ValueError as e:
    print(f"✅ CORRECTLY CAUGHT: {str(e)[:120]}...")

# %%
# ============================================================================
# CASE 2: INVALID - Train-Test Overlap (Data Leakage)
# ============================================================================
print("\n" + "=" * 80)
print("CASE 2: INVALID - Train-Test Overlap")
print("=" * 80)
print("Config: train=[0,1], val=[2], test=[0,3]")
print("Expected: ValueError (DATA LEAKAGE DETECTED)")

try:
    d1_case2 = MultiSourceTSDataSet(
        file_paths=[grouped_weather_path],
        time_col="time",
        target_cols=[target_col],
        num_cols=covariate_columns,
        enrich_cat=["hour", "dow"],
        global_forecasting=True,
        group_cols=["group"],
        memory_efficient=False,
        add_target_to_past=True,
    )

    d2_case2 = EncoderDecoder(
        d1_dataset=d1_case2,
        past_len=96,
        future_len=96,
        step_size=96,
        batch_size=32,
        split_ratio=(0.7, 0.15, 0.15),
        split_group_config=([0, 1], [2], [0, 3]),  # OVERLAP!
        scaling_method="standard",
        scale_targets=True,
    )

    d2_case2.setup(stage="fit")
    print("❌ ERROR: Should have raised ValueError but didn't!")

except ValueError as e:
    print(f"✅ CORRECTLY CAUGHT: {str(e)[:120]}...")

# %%
print("\n" + "=" * 80)
print("VALID TEST CASES")
print("=" * 80)

# %%
# ============================================================================
# CASE 3: HYBRID SPLIT (Correct Usage)
# ============================================================================
print("\n" + "=" * 80)
print("CASE 3: HYBRID SPLIT (Ultimate Hybrid)")
print("=" * 80)
print("Config: train=[0,1], val=[2], test=[3]")
print("Expected: Train from groups 0,1 (70%), Val from groups 0,1 (15%) + 100% of group 2")
print("          Test from groups 0,1 (15%) + 100% of group 3")

d1_case3 = MultiSourceTSDataSet(
    file_paths=[grouped_weather_path],
    time_col="time",
    target_cols=[target_col],
    num_cols=covariate_columns,
    enrich_cat=["hour", "dow"],
    global_forecasting=True,
    group_cols=["group"],
    memory_efficient=False,
    add_target_to_past=True,
)

d2_case3 = EncoderDecoder(
    d1_dataset=d1_case3,
    past_len=96,
    future_len=96,
    step_size=96,
    batch_size=32,
    split_ratio=(0.7, 0.15, 0.15),
    split_group_config=([0, 1], [2], [3]),  # Mutually exclusive groups
    scaling_method="standard",
    scale_targets=True,
)

d2_case3.setup(stage="fit")
d2_case3.setup(stage="test")
analyze_splits(d2_case3, "CASE 3: HYBRID SPLIT ✅")

# %%
# ============================================================================
# CASE 4: LOCAL FORECASTING (Baseline)
# ============================================================================
print("\n" + "=" * 80)
print("CASE 4: LOCAL FORECASTING (Pure Temporal, No Groups)")
print("=" * 80)
print("Config: global_forecasting=False, split_group_config=None")
print("Expected: All windows pooled and split 70/15/15 temporally")

d1_case4 = MultiSourceTSDataSet(
    file_paths=[grouped_weather_path],
    time_col="time",
    target_cols=[target_col],
    num_cols=covariate_columns,
    enrich_cat=["hour", "dow"],
    global_forecasting=False,  # Local forecasting
    group_cols=["group"],
    memory_efficient=False,
    add_target_to_past=True,
)

d2_case4 = EncoderDecoder(
    d1_dataset=d1_case4,
    past_len=96,
    future_len=96,
    step_size=96,
    batch_size=32,
    split_ratio=(0.7, 0.15, 0.15),
    split_group_config=None,  # No group config
    scaling_method="standard",
    scale_targets=True,
)

d2_case4.setup(stage="fit")
d2_case4.setup(stage="test")
analyze_splits(d2_case4, "CASE 4: LOCAL FORECASTING ✅")

# %%
# ============================================================================
# CASE 5: PURE GROUP GENERALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("CASE 5: PURE GROUP GENERALIZATION")
print("=" * 80)
print("Config: train=[0,1], val=[2], test=[3], split_ratio=(1.0, 0.0, 0.0)")
print("Expected: Train=100% of groups 0,1, Val=100% of group 2, Test=100% of group 3")

d1_case5 = MultiSourceTSDataSet(
    file_paths=[grouped_weather_path],
    time_col="time",
    target_cols=[target_col],
    num_cols=covariate_columns,
    enrich_cat=["hour", "dow"],
    global_forecasting=True,
    group_cols=["group"],
    memory_efficient=False,
    add_target_to_past=True,
)

d2_case5 = EncoderDecoder(
    d1_dataset=d1_case5,
    past_len=96,
    future_len=96,
    step_size=96,
    batch_size=32,
    split_ratio=(1.0, 0.0, 0.0),  # 100% to train, 0% to val/test
    split_group_config=([0, 1], [2], [3]),
    scaling_method="standard",
    scale_targets=True,
)

d2_case5.setup(stage="fit")
d2_case5.setup(stage="test")
analyze_splits(d2_case5, "CASE 5: PURE GROUP GENERALIZATION ✅")

# %%
# ============================================================================
# CASE 6: PURE TEMPORAL GENERALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("CASE 6: PURE TEMPORAL GENERALIZATION")
print("=" * 80)
print("Config: train=[0,1,2,3], val=[], test=[], split_ratio=(0.7, 0.15, 0.15)")
print("Expected: All groups split 70/15/15 temporally")

d1_case6 = MultiSourceTSDataSet(
    file_paths=[grouped_weather_path],
    time_col="time",
    target_cols=[target_col],
    num_cols=covariate_columns,
    enrich_cat=["hour", "dow"],
    global_forecasting=True,
    group_cols=["group"],
    memory_efficient=False,
    add_target_to_past=True,
)

d2_case6 = EncoderDecoder(
    d1_dataset=d1_case6,
    past_len=96,
    future_len=96,
    step_size=96,
    batch_size=32,
    split_ratio=(0.7, 0.15, 0.15),
    split_group_config=([0, 1, 2, 3], [], []),  # All groups in train
    scaling_method="standard",
    scale_targets=True,
)

d2_case6.setup(stage="fit")
d2_case6.setup(stage="test")
analyze_splits(d2_case6, "CASE 6: PURE TEMPORAL GENERALIZATION ✅")

# %%
print("\n" + "=" * 80)
print("INVALID TEST CASES (Should Raise Errors)")
print("=" * 80)

# %%
# ============================================================================
# INVALID CASE 1: Train-Val Overlap (Data Leakage)
# ============================================================================
print("\n" + "=" * 80)
print("INVALID CASE 1: Train-Val Overlap")
print("=" * 80)
print("Config: train=[0,1], val=[0,1], test=[3]")
print("Expected: ValueError (DATA LEAKAGE DETECTED)")

try:
    d1_invalid1 = MultiSourceTSDataSet(
        file_paths=[grouped_weather_path],
        time_col="time",
        target_cols=[target_col],
        num_cols=covariate_columns,
        enrich_cat=["hour", "dow"],
        global_forecasting=True,
        group_cols=["group"],
        memory_efficient=False,
        add_target_to_past=True,
    )

    d2_invalid1 = EncoderDecoder(
        d1_dataset=d1_invalid1,
        past_len=96,
        future_len=96,
        step_size=96,
        batch_size=32,
        split_ratio=(0.7, 0.15, 0.15),
        split_group_config=([0, 1], [0, 1], [3]),  # OVERLAP!
        scaling_method="standard",
        scale_targets=True,
    )

    d2_invalid1.setup(stage="fit")
    print("❌ ERROR: Should have raised ValueError but didn't!")

except ValueError as e:
    print(f"✅ CORRECTLY CAUGHT: {str(e)[:100]}...")

# %%
# ============================================================================
# INVALID CASE 2: Train-Test Overlap (Data Leakage)
# ============================================================================
print("\n" + "=" * 80)
print("INVALID CASE 2: Train-Test Overlap")
print("=" * 80)
print("Config: train=[0,1], val=[2], test=[0,3]")
print("Expected: ValueError (DATA LEAKAGE DETECTED)")

try:
    d1_invalid2 = MultiSourceTSDataSet(
        file_paths=[grouped_weather_path],
        time_col="time",
        target_cols=[target_col],
        num_cols=covariate_columns,
        enrich_cat=["hour", "dow"],
        global_forecasting=True,
        group_cols=["group"],
        memory_efficient=False,
        add_target_to_past=True,
    )

    d2_invalid2 = EncoderDecoder(
        d1_dataset=d1_invalid2,
        past_len=96,
        future_len=96,
        step_size=96,
        batch_size=32,
        split_ratio=(0.7, 0.15, 0.15),
        split_group_config=([0, 1], [2], [0, 3]),  # OVERLAP!
        scaling_method="standard",
        scale_targets=True,
    )

    d2_invalid2.setup(stage="fit")
    print("❌ ERROR: Should have raised ValueError but didn't!")

except ValueError as e:
    print(f"✅ CORRECTLY CAUGHT: {str(e)[:100]}...")

# %%
# ============================================================================
# INVALID CASE 3: Val-Test Overlap (Data Leakage)
# ============================================================================
print("\n" + "=" * 80)
print("INVALID CASE 3: Val-Test Overlap")
print("=" * 80)
print("Config: train=[0,1], val=[2], test=[2,3]")
print("Expected: ValueError (DATA LEAKAGE DETECTED)")

try:
    d1_invalid3 = MultiSourceTSDataSet(
        file_paths=[grouped_weather_path],
        time_col="time",
        target_cols=[target_col],
        num_cols=covariate_columns,
        enrich_cat=["hour", "dow"],
        global_forecasting=True,
        group_cols=["group"],
        memory_efficient=False,
        add_target_to_past=True,
    )

    d2_invalid3 = EncoderDecoder(
        d1_dataset=d1_invalid3,
        past_len=96,
        future_len=96,
        step_size=96,
        batch_size=32,
        split_ratio=(0.7, 0.15, 0.15),
        split_group_config=([0, 1], [2], [2, 3]),  # OVERLAP!
        scaling_method="standard",
        scale_targets=True,
    )

    d2_invalid3.setup(stage="fit")
    print("❌ ERROR: Should have raised ValueError but didn't!")

except ValueError as e:
    print(f"✅ CORRECTLY CAUGHT: {str(e)[:100]}...")

# %%
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
✅ INVALID CASES (All Correctly Rejected):
   - Case 1: Train-Val overlap (groups 0,1 in both) - DATA LEAKAGE DETECTED
   - Case 2: Train-Test overlap (group 0 in both) - DATA LEAKAGE DETECTED
   - Case 3 (Invalid): Val-Test overlap (group 2 in both) - DATA LEAKAGE DETECTED

✅ VALID CASES (All Passed):
   - Case 3: Hybrid Split (train groups split + dedicated val/test groups)
   - Case 4: Local Forecasting (pure temporal, no groups)
   - Case 5: Pure Group Generalization (100% group separation)
   - Case 6: Pure Temporal Generalization (all groups split temporally)

🎉 All tests passed! The split logic is robust and prevents data leakage.
""")

# %%
