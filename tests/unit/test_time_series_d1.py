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
                        'cat_feature': f'cat_{np.random.randint(0, 3)}'
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
        'cat_cols': ['cat_feature']
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
        )
        
        # Check that the dataset was initialized correctly
        assert len(d1_dataset.file_paths) == 2
        assert d1_dataset.time_col == 'time'
        assert d1_dataset.group_cols == ['group']
        assert d1_dataset.feature_cols == ['feature_0', 'feature_1']
        assert d1_dataset.target_cols == ['target_0']
        assert d1_dataset.cat_cols == ['cat_feature']
        assert not d1_dataset.memory_efficient
        # Get data for the first group
        group_id = 0
        group_data = d1_dataset._load_group_data(group_id)
        
        # Check that the returned data has the expected format
        self.assertTrue(isinstance(group_data, pd.DataFrame))
        self.assertTrue(self.time_col in group_data.columns)
        for col in self.feature_cols:
            self.assertTrue(col in group_data.columns)
        for col in self.target_cols:
            self.assertTrue(col in group_data.columns)
    
    def test_get_group_data(self):
        """Test _get_group_data method."""
        d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            memory_efficient=True  # Test with memory efficient mode
        )
        
        # Get data for the first group
        group_id = 0
        group_data = d1_dataset._get_group_data(group_id)
        
        # Check that the returned data has the expected format
        self.assertTrue(isinstance(group_data, pd.DataFrame))
        self.assertTrue(self.time_col in group_data.columns)
        for col in self.feature_cols:
            self.assertTrue(col in group_data.columns)
        for col in self.target_cols:
            self.assertTrue(col in group_data.columns)
        
        # Test caching behavior
        # First call should load from disk
        d1_dataset.data_cache = {}  # Clear cache
        group_data1 = d1_dataset._get_group_data(group_id)
        
        # Second call should use cache
        group_data2 = d1_dataset._get_group_data(group_id)
        
        # Both should be the same data
        pd.testing.assert_frame_equal(group_data1, group_data2)
    
    def test_len(self):
        """Test __len__ method."""
        d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            memory_efficient=False
        )
        
        # Check that the length is correct
        # We should have 3 groups total (some in file 0, some in file 1)
        self.assertEqual(len(d1_dataset), 3)
    
    def test_static_cols(self):
        """Test specifying static columns."""
        # Add a static column to the test data
        for file_idx in range(2):
            df = pd.read_csv(os.path.join(self.temp_dir, f'test_data_{file_idx}.csv'))
            df['static_feature'] = df['group'].apply(lambda x: float(x.split('_')[1]) * 10)
            df.to_csv(os.path.join(self.temp_dir, f'test_data_{file_idx}.csv'), index=False)
        
        # Create dataset with static column
        static_cols = ['static_feature']
        d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            static_cols=static_cols,
            memory_efficient=False
        )
        
        # Check that static columns were set correctly
        self.assertListEqual(d1_dataset.static_cols, static_cols)
        
        # Get data for the first group
        group_data = d1_dataset[0]
        
        # Check that static features are included
        self.assertTrue('st' in group_data)
        self.assertEqual(len(group_data['st']), len(static_cols))


if __name__ == '__main__':
    unittest.main()
