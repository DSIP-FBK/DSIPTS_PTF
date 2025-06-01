import unittest
import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dsipts.data_structure.time_series_d1 import MultiSourceTSDataSet


class TestMultiSourceTSDataSet(unittest.TestCase):
    """Test cases for the MultiSourceTSDataSet class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV files
        self.create_test_files()
        
        # Define common parameters
        self.file_paths = [
            os.path.join(self.temp_dir, f'test_data_{i}.csv') for i in range(2)
        ]
        self.group_cols = 'group'
        self.time_col = 'time'
        self.feature_cols = ['feature_0', 'feature_1']
        self.target_cols = ['target_0']
        self.cat_cols = ['cat_feature']
        
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test CSV files."""
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
            df.to_csv(os.path.join(self.temp_dir, f'test_data_{file_idx}.csv'), index=False)
    
    def test_init_memory_efficient_false(self):
        """Test initialization with memory_efficient=False."""
        d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            memory_efficient=False
        )
        
        # Check that the dataset was initialized correctly
        self.assertEqual(len(d1_dataset.file_paths), 2)
        self.assertEqual(d1_dataset.time_col, 'time')
        self.assertEqual(d1_dataset.group_cols, ['group'])
        self.assertListEqual(d1_dataset.feature_cols, ['feature_0', 'feature_1'])
        self.assertListEqual(d1_dataset.target_cols, ['target_0'])
        self.assertListEqual(d1_dataset.cat_cols, ['cat_feature'])
        self.assertFalse(d1_dataset.memory_efficient)
        
        # Check that data was preloaded
        self.assertTrue(len(d1_dataset.data_cache) > 0)
        
        # Check that metadata was created
        self.assertTrue('cols' in d1_dataset.metadata)
        self.assertTrue('max_classes' in d1_dataset.metadata)
        self.assertTrue('known_cat_cols' in d1_dataset.metadata)
        self.assertTrue('known_num_cols' in d1_dataset.metadata)
    
    def test_init_memory_efficient_true(self):
        """Test initialization with memory_efficient=True."""
        d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            memory_efficient=True
        )
        
        # Check that the dataset was initialized correctly
        self.assertEqual(len(d1_dataset.file_paths), 2)
        self.assertEqual(d1_dataset.time_col, 'time')
        self.assertEqual(d1_dataset.group_cols, ['group'])
        self.assertListEqual(d1_dataset.feature_cols, ['feature_0', 'feature_1'])
        self.assertListEqual(d1_dataset.target_cols, ['target_0'])
        self.assertListEqual(d1_dataset.cat_cols, ['cat_feature'])
        self.assertTrue(d1_dataset.memory_efficient)
        
        # Check that data was not preloaded
        self.assertEqual(len(d1_dataset.data_cache), 0)
    
    def test_getitem(self):
        """Test __getitem__ method."""
        d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            memory_efficient=False
        )
        
        # Get data for the first group
        group_data = d1_dataset[0]
        
        # Check that the returned data has the expected format
        self.assertTrue('x' in group_data)
        self.assertTrue('y' in group_data)
        self.assertTrue('t' in group_data)
        self.assertTrue('group_id' in group_data)
        
        # Check dimensions
        self.assertEqual(group_data['x'].shape[1], len(self.feature_cols))
        self.assertEqual(group_data['y'].shape[1], len(self.target_cols))
        self.assertEqual(len(group_data['t']), len(group_data['x']))
    
    def test_known_unknown_cols(self):
        """Test specifying known and unknown columns."""
        # Specify custom known and unknown columns
        known_cols = ['feature_0']
        unknown_cols = ['feature_1', 'target_0']
        
        d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            known_cols=known_cols,
            unknown_cols=unknown_cols,
            memory_efficient=False
        )
        
        # Check that known and unknown columns were set correctly
        self.assertListEqual(d1_dataset.known_cols, known_cols)
        self.assertListEqual(d1_dataset.unknown_cols, unknown_cols)
        
        # Check metadata
        self.assertListEqual(d1_dataset.metadata['known_cols'], known_cols)
        self.assertListEqual(d1_dataset.metadata['unknown_cols'], unknown_cols)
        self.assertListEqual(d1_dataset.metadata['known_num_cols'], known_cols)  # feature_0 is numerical
        self.assertEqual(len(d1_dataset.metadata['known_cat_cols']), 0)  # No categorical known cols
    
    def test_load_group_data(self):
        """Test _load_group_data method."""
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
