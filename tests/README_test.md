# Testing Manual

This directory contains all tests for the DSIPTS library with comprehensive validation and strong assertions.

## Test Folder Structure

```
tests/
├── unit/
│   ├── test_time_series_d1.py  # D1 layer (MultiSourceTSDataSet) unit tests
│   └── test_time_series_d2.py  # D2 layer (EncoderDecoder) unit tests
└── integration/
    ├── test_d1_d2_integration.py        # D1-D2 pipeline integration tests
    ├── test_encoder_decoder_scaler.py   # Scaling functionality tests
    └── test_d1_d2_analysis.py          # Analysis tool for manual inspection
```

## 📊 Test Coverage Summary

| Category | Tests | Status |
|----------|-------|--------|
| **Unit Tests** | 43 | ✅ All passing |
| **Integration Tests** | 11 | ✅ All passing |
| **Analysis Tool** | 1 | ✅ Diagnostic tool |
| **Total** | **54** | **✅ 100%** |

---

## Unit Tests

### test_time_series_d1.py (18 tests)
Tests for D1 layer (`MultiSourceTSDataSet`):

**Core Functionality:**
- ✅ Basic initialization and data loading
- ✅ Memory-efficient vs in-memory modes
- ✅ Multi-file and DataFrame inputs
- ✅ Global vs local forecasting

**Feature Handling:**
- ✅ Temporal enrichment (hour, dow, month, minute)
- ✅ Categorical encoding with ordered lists
- ✅ past_cols/future_cols configuration
- ✅ Edge cases (insufficient data, empty groups)

**⭐ COMPREHENSIVE VALIDATION (NEW):**
- ✅ **Temporal cardinalities validation** - Verifies hour=24, dow=7, month=12, minute=60
- ✅ **All temporal features cardinalities** - Validates all temporal features have correct values
- ✅ **Strong assertions** - Tests actual behavior, not just execution

### test_time_series_d2.py (25 tests)
Tests for D2 layer (`EncoderDecoder`):

**Core Functionality:**
- ✅ Window creation and step size
- ✅ Batch structure and shapes
- ✅ Data splitting (temporal, random)
- ✅ DataLoader integration

**Scaling:**
- ✅ StandardScaler integration
- ✅ MinMaxScaler integration
- ✅ Target scaling (enabled/disabled)

**⭐ COMPREHENSIVE VALIDATION (NEW):**
- ✅ **StandardScaler validation** - Verifies mean≈0, std≈1 on actual data
- ✅ **MinMaxScaler validation** - Verifies values in [0, 1] range
- ✅ **Memory-efficient scaling** - Tests scaling with chunked data loading
- ✅ **D1→D2 integration** - Validates temporal features propagate correctly
- ✅ **Strong assertions** - Tests actual transformed values, not just shapes

**Advanced Features:**
- ✅ Temporal feature propagation
- ✅ Global vs local forecasting
- ✅ Window boundary edge cases

---

## Integration Tests

### test_d1_d2_integration.py (8 tests)
End-to-end D1→D2 pipeline tests:

**Pipeline Tests:**
- ✅ Basic D1→D2 pipeline
- ✅ Pipeline with scaling
- ✅ DataLoader integration

**Scaling Tests:**
- ✅ Scaling isolation (train/val/test)
- ✅ Inverse scaling accuracy
- ✅ Memory-efficient mode with scaling

**Feature Tests:**
- ✅ Temporal enrichment propagation
- ✅ Multiple targets handling

### test_encoder_decoder_scaler.py (3 tests)
Detailed scaling functionality tests:

- ✅ StandardScaler (mean=0, std=1)
- ✅ MinMaxScaler (range [0,1])
- ✅ Target scaling

### test_d1_d2_analysis.py (Analysis Tool)
**Purpose:** Diagnostic tool for manual log inspection (NOT a pass/fail test)

**What It Does:**
- Generates comprehensive logs for all D1/D2 scenarios
- Tests 5 major categories: data loading, global/local forecasting, temporal enrichment, scaling, comprehensive scenarios
- Provides detailed metadata, shapes, statistics for manual validation

**When to Use:**
- ✅ Before releases - Generate logs and manually inspect
- ✅ When debugging - Compare actual vs expected behavior
- ✅ When adding features - Validate complete data flow
- ✅ When behavior seems wrong - Detailed visibility into all layers

**How to Use:**
```bash
# Generate analysis logs
python tests/integration/test_d1_d2_analysis.py > analysis.log 2>&1

# Inspect specific sections
grep "cardinalities" analysis.log
grep "Scaled Feature Statistics" analysis.log
```

**Why Keep It:**
This tool caught 2 critical bugs that automated tests missed:
1. Temporal cardinalities were wrong ([2, 3] instead of [2, 3, 24, 7])
2. Scaling statistics showed unscaled data (mean=20-60 instead of ≈0)

**Automated tests passed but were too weak** - they only checked execution, not actual values. This analysis tool provides the comprehensive visibility needed to catch such issues.

---

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suites
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_time_series_d1.py -v

# Specific test class or function
pytest tests/unit/test_time_series_d2.py::TestD2Scaling -v
pytest tests/integration/test_d1_d2_integration.py::TestD1D2Pipeline::test_basic_pipeline -v
```

### Run with Coverage
```bash
pytest tests/ --cov=dsipts --cov-report=html
```

---

## Test Organization Principles

### Unit Tests (`tests/unit/`)
- **Fast** - Each test runs in milliseconds
- **Isolated** - Tests one specific function/method
- **Focused** - Clear, single responsibility
- **No external dependencies** - Uses fixtures and mocks

### Integration Tests (`tests/integration/`)
- **End-to-end** - Tests complete workflows
- **Real scenarios** - Uses actual data pipelines
- **Cross-component** - Tests D1→D2 interactions
- **Performance** - May take longer to run

---

## Testing Philosophy: Strong Assertions

### The Problem We Solved
**Passing tests ≠ Correct behavior**

Previous tests were too weak:
- ❌ Only checked that code ran without errors
- ❌ Didn't validate actual values
- ❌ Missed critical bugs that manual log inspection caught

### Our Solution: Comprehensive Validation

**Before (Weak Tests):**
```python
# Just checks it runs
assert len(d2.valid_windows) > 0
assert d2.is_scaler_fitted
```

**After (Strong Tests):**
```python
# Validates actual behavior
assert 24 in cardinalities, "hour cardinality (24) not found"
assert abs(mean) < 0.5, f"Feature mean {mean} not close to 0"
assert 0.8 < std < 1.2, f"Feature std {std} not close to 1"
```

### Two-Tier Approach

**1. Automated Tests (CI/CD)**
- Strong assertions on actual values
- Must pass for every commit
- Fast feedback (seconds)

**2. Analysis Tool (Manual Inspection)**
- Comprehensive logs for all scenarios
- Used before releases and for debugging
- Catches subtle issues automated tests might miss

### Key Improvements Made

✅ **Temporal Cardinalities** - Now validates hour=24, dow=7, month=12, minute=60
✅ **Scaling Validation** - Verifies StandardScaler produces mean≈0, std≈1
✅ **MinMaxScaler Validation** - Verifies values in [0, 1] range
✅ **Memory-Efficient Mode** - Tests scaling with chunked data loading
✅ **D1→D2 Integration** - Validates temporal features propagate correctly

---

## Key Changes from Previous Version

### ✅ Improvements
1. **Added comprehensive validation tests** - Strong assertions on actual values
2. **Merged validation into D1/D2 tests** - Better organization, no duplicate files
3. **Added analysis tool** - Diagnostic tool for manual inspection
4. **Removed obsolete tests** - D1 no longer handles scaling
5. **Cleaner structure** - 2 unit test files, 3 integration test files (including analysis tool)
6. **All tests passing** - 54/54 tests ✅

### ⭐ New Tests Added
- **D1: Temporal cardinalities validation** - Verifies hour=24, dow=7, month=12, minute=60
- **D1: All temporal features cardinalities** - Validates all temporal features
- **D2: StandardScaler validation** - Verifies mean≈0, std≈1
- **D2: MinMaxScaler validation** - Verifies values in [0, 1]
- **D2: Memory-efficient scaling** - Tests scaling with chunked data
- **D2: D1→D2 integration** - Validates temporal feature propagation

### ❌ Removed Files
- `test_comprehensive_validation.py` - Merged into D1/D2 unit tests
- `TESTING_STRATEGY.md` - Consolidated into README_test.md

---

## Requirements
- Python 3.8+
- pytest >= 7.0
- All dependencies from `requirements.txt`
- Temporary test files are auto-cleaned

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Make sure you're in the project root
cd /path/to/DSIPTS_PTF
pytest tests/
```

**Fixture not found:**
- Check that fixtures are defined in the same file or in `conftest.py`

**Tests hanging:**
- Check for infinite loops or missing test data cleanup

**Memory issues:**
- Run tests individually: `pytest tests/unit/test_time_series_d1.py`

---

## Contributing Tests

When adding new tests:
1. **Unit tests** - Add to appropriate file in `tests/unit/`
2. **Integration tests** - Add to appropriate file in `tests/integration/`
3. **Follow naming convention** - `test_<feature_name>`
4. **Add docstrings** - Explain what the test validates
5. **Update this README** - Document new test coverage

---

For more details on the library, see the main [README.md](../README.md).
