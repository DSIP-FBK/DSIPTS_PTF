import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
import pytest

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dsipts.data_structure.time_series_d1 import MultiSourceTSDataSet, extend_time_df


@pytest.fixture
def test_data():
    """Fixture to create test data for D1 layer tests."""
    # Create a temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    
    # Generate two CSV files with different groups
    for file_idx in range(2):
        data = []
        
        # Generate data for each group
        for group_idx in range(3):
            # Determine which file gets which groups
            if (group_idx % 2 == 0 and file_idx == 0) or (group_idx % 2 == 1 and file_idx == 1):
                # Generate time series for this group
                for t in range(10):
                    row = {
                        'group': f'group_{group_idx}',
                        'time': t,
                        'feature_0': np.sin(t/10 + group_idx) + np.random.normal(0, 0.1),
                        'feature_1': np.cos(t/10 + group_idx) + np.random.normal(0, 0.1),
                        'target_0': np.sin(t/5 + group_idx) + np.random.normal(0, 0.1),
                        'cat_feature': f'cat_{np.random.randint(0, 3)}',
                        'static_feature': float(group_idx) * 10
                    }
                    data.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(temp_dir, f'test_data_{file_idx}.csv'), index=False)
    
    # Define common parameters
    params = {
        'file_paths': [os.path.join(temp_dir, f'test_data_{i}.csv') for i in range(2)],
        'group_cols': 'group',
        'time_col': 'time',
        'feature_cols': ['feature_0', 'feature_1'],
        'target_cols': ['target_0'],
        'cat_cols': ['cat_feature'],
        'static_cols': ['static_feature']
    }
    
    # Yield the test data parameters
    yield params
    
    # Clean up temporary files
    shutil.rmtree(temp_dir)
    
def test_init_memory_efficient_false(test_data):
    """Test initialization with memory_efficient=False."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data['file_paths'],
        group_cols=test_data['group_cols'],
        time_col=test_data['time_col'],
        feature_cols=test_data['feature_cols'],
        target_cols=test_data['target_cols'],
        cat_cols=test_data['cat_cols'],
        memory_efficient=False
    )
    
    # Check that the dataset was initialized correctly
    assert len(d1_dataset.file_paths) == 2
    assert d1_dataset.time_col == 'time'
    assert d1_dataset.group_cols == ['group']
    assert d1_dataset.feature_cols == ['feature_0', 'feature_1']
    assert d1_dataset.target_cols == ['target_0']
    assert d1_dataset.cat_cols == ['cat_feature']
    assert not d1_dataset.memory_efficient
    
    # Check that data was preloaded
    assert len(d1_dataset.data_cache) > 0
    
    # Check that metadata was created
    assert 'cols' in d1_dataset.metadata
    assert 'max_classes' in d1_dataset.metadata
    assert 'known_cat_cols' in d1_dataset.metadata
    assert 'known_num_cols' in d1_dataset.metadata
    
def test_init_memory_efficient_true(test_data):
    """Test initialization with memory_efficient=True."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data['file_paths'],
        group_cols=test_data['group_cols'],
        time_col=test_data['time_col'],
        feature_cols=test_data['feature_cols'],
        target_cols=test_data['target_cols'],
        cat_cols=test_data['cat_cols'],
        memory_efficient=True
    )
    
    # Check that the dataset was initialized correctly
    assert len(d1_dataset.file_paths) == 2
    assert d1_dataset.time_col == 'time'
    assert d1_dataset.group_cols == ['group']
    assert d1_dataset.feature_cols == ['feature_0', 'feature_1']
    assert d1_dataset.target_cols == ['target_0']
    assert d1_dataset.cat_cols == ['cat_feature']
    assert d1_dataset.memory_efficient
    
    # Check that data was not preloaded
    assert len(d1_dataset.data_cache) == 0
    
def test_getitem(test_data):
    """Test __getitem__ method."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data['file_paths'],
        group_cols=test_data['group_cols'],
        time_col=test_data['time_col'],
        feature_cols=test_data['feature_cols'],
        target_cols=test_data['target_cols'],
        cat_cols=test_data['cat_cols'],
        memory_efficient=False
    )
    
    # Get data for the first group
    group_data = d1_dataset[0]
    
    # Check that the returned data has the expected format
    assert 'x' in group_data
    assert 'y' in group_data
    assert 't' in group_data
    assert 'group_id' in group_data
    
    # Check dimensions
    assert group_data['x'].shape[1] == len(test_data['feature_cols'])
    assert group_data['y'].shape[1] == len(test_data['target_cols'])
    assert len(group_data['t']) == len(group_data['x'])
    
def test_known_unknown_cols(test_data):
    """Test specifying known and unknown columns."""
    # Create dataset with custom known/unknown columns
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data['file_paths'],
        group_cols=test_data['group_cols'],
        time_col=test_data['time_col'],
        feature_cols=test_data['feature_cols'],
        target_cols=test_data['target_cols'],
        cat_cols=test_data['cat_cols'],
        known_cols=['feature_0'],  # Only feature_0 is known at prediction time
        unknown_cols=['feature_1', 'target_0']  # feature_1 and target_0 are unknown
    )
    
    # Check that the columns were correctly categorized
    assert d1_dataset.known_cols == ['feature_0']
    assert d1_dataset.unknown_cols == ['feature_1', 'target_0']
    
    # Check metadata
    metadata = d1_dataset.get_metadata()
    assert metadata['known_cols'] == ['feature_0']
    assert metadata['unknown_cols'] == ['feature_1', 'target_0']


def test_get_group_data(test_data):
    """Test _get_group_data method."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data['file_paths'],
        group_cols=test_data['group_cols'],
        time_col=test_data['time_col'],
        feature_cols=test_data['feature_cols'],
        target_cols=test_data['target_cols'],
        cat_cols=test_data['cat_cols'],
        memory_efficient=True  # Test with memory efficient mode
    )
    
    # Get data for the first group
    group_id = 0
    group_data = d1_dataset._get_group_data(group_id)
    
    # Check that the returned data has the expected format
    assert isinstance(group_data, pd.DataFrame)
    assert test_data['time_col'] in group_data.columns
    for col in test_data['feature_cols']:
        assert col in group_data.columns
    for col in test_data['target_cols']:
        assert col in group_data.columns
    
    # Test caching behavior
    # First call should load from disk
    d1_dataset.data_cache = {}  # Clear cache
    group_data1 = d1_dataset._get_group_data(group_id)
        
    # Second call should use cache
    group_data2 = d1_dataset._get_group_data(group_id)
    
    # Both should be identical
    assert np.array_equal(group_data1.values, group_data2.values)


def test_len(test_data):
    """Test __len__ method."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data['file_paths'],
        group_cols=test_data['group_cols'],
        time_col=test_data['time_col'],
        feature_cols=test_data['feature_cols'],
        target_cols=test_data['target_cols'],
        cat_cols=test_data['cat_cols'],
        memory_efficient=False
    )
    
    # Check that the length is correct
    # We should have 3 groups total (some in file 0, some in file 1)
    assert len(d1_dataset) == 3


def test_static_cols(test_data):
    """Test handling of static columns."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data['file_paths'],
        group_cols=test_data['group_cols'],
        time_col=test_data['time_col'],
        feature_cols=test_data['feature_cols'],
        target_cols=test_data['target_cols'],
        cat_cols=test_data['cat_cols'],
        static_cols=['static_feature'],
        memory_efficient=False
    )
    
    # Check that static columns were set correctly
    assert d1_dataset.static_cols == ['static_feature']
    
    # Get data for the first group
    group_data = d1_dataset[0]
    
    # Check that static features are included
    assert 'st' in group_data
    assert len(group_data['st']) == 1  # One static feature


def test_data_caching(test_data):
    """Test data caching behavior."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data['file_paths'],
        group_cols=test_data['group_cols'],
        time_col=test_data['time_col'],
        feature_cols=test_data['feature_cols'],
        target_cols=test_data['target_cols'],
        cat_cols=test_data['cat_cols'],
        memory_efficient=False
    )
    
    # First access should load data
    group_data1 = d1_dataset[0]
    
    # Second access should use cache
    group_data2 = d1_dataset[0]
    
    # Both should be identical
    assert np.array_equal(group_data1['x'], group_data2['x'])
    assert np.array_equal(group_data1['y'], group_data2['y'])
    assert np.array_equal(group_data1['t'], group_data2['t'])
    
    # Test internal cache
    d1_dataset.data_cache = {}  # Clear cache
    group_data3 = d1_dataset._get_group_data(0)
    group_data4 = d1_dataset._get_group_data(0)
    assert np.array_equal(group_data3.values, group_data4.values)


def test_extend_time_df():
    """Test the extend_time_df function."""
    # Create sample data with gaps
    df = pd.DataFrame({
        'time': [0, 2, 4],
        'value': [1.0, 2.0, 3.0],
        'group': ['A', 'A', 'A']
    })
    
    # Extend the time series
    extended_df = extend_time_df(
        df=df,
        time_col='time',
        freq=1,
        group_cols=['group']
    )
    
    # Check that gaps were filled
    assert len(extended_df) == 5  # Should now have rows for t=0,1,2,3,4
    assert list(extended_df['time']) == [0, 1, 2, 3, 4]
    assert np.isnan(extended_df.loc[1, 'value'])  # t=1 should be NaN
    assert np.isnan(extended_df.loc[3, 'value'])  # t=3 should be NaN
