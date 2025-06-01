import unittest
import os
import sys
import pandas as pd
import numpy as np
import torch
import tempfile
import shutil
from torch.utils.data import DataLoader

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dsipts.data_structure.time_series_d1 import MultiSourceTSDataSet
from dsipts.data_structure.time_series_d2 import TSDataModule, TimeSeriesSubset, custom_collate_fn


class TestTSDataModule(unittest.TestCase):
    """Test cases for the TSDataModule class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV files
        self.create_test_files()
        
        # Create D1 dataset
        self.file_paths = [
            os.path.join(self.temp_dir, f'test_data_{i}.csv') for i in range(2)
        ]
        self.group_cols = 'group'
        self.time_col = 'time'
        self.feature_cols = ['feature_0', 'feature_1']
        self.target_cols = ['target_0']
        self.cat_cols = ['cat_feature']
        
        self.d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            memory_efficient=False
        )
        
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
                    for t in range(20):  # Longer sequences for window creation
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
    
    def test_init_percentage_split(self):
        """Test initialization with percentage split."""
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
            memory_efficient=False
        )
        
        # Check that the module was initialized correctly
        self.assertEqual(d2_module.past_len, 5)
        self.assertEqual(d2_module.future_len, 2)
        self.assertEqual(d2_module.batch_size, 32)
        self.assertEqual(d2_module.min_valid_length, 4)
        self.assertEqual(d2_module.split_method, 'percentage')
        self.assertEqual(d2_module.split_config, (0.7, 0.15, 0.15))
        
        # Check that splits were created
        self.assertTrue(hasattr(d2_module, 'train_indices'))
        self.assertTrue(hasattr(d2_module, 'val_indices'))
        self.assertTrue(hasattr(d2_module, 'test_indices'))
        
        # Check that the sum of split sizes equals the total number of windows
        total_windows = len(d2_module.mapping)
        split_sum = len(d2_module.train_indices) + len(d2_module.val_indices) + len(d2_module.test_indices)
        self.assertEqual(split_sum, total_windows)
        
        # Check approximate split ratios (allowing for rounding)
        self.assertAlmostEqual(len(d2_module.train_indices) / total_windows, 0.7, delta=0.05)
        self.assertAlmostEqual(len(d2_module.val_indices) / total_windows, 0.15, delta=0.05)
        self.assertAlmostEqual(len(d2_module.test_indices) / total_windows, 0.15, delta=0.05)
    
    def test_init_group_split(self):
        """Test initialization with group split."""
        # Get all group indices
        all_groups = list(range(len(self.d1_dataset)))
        
        # Split groups into train/val/test
        train_groups = all_groups[:int(0.7 * len(all_groups))]
        val_groups = all_groups[int(0.7 * len(all_groups)):int(0.85 * len(all_groups))]
        test_groups = all_groups[int(0.85 * len(all_groups)):]
        
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='group',
            split_config=(train_groups, val_groups, test_groups),
            memory_efficient=False
        )
        
        # Check that the module was initialized correctly
        self.assertEqual(d2_module.past_len, 5)
        self.assertEqual(d2_module.future_len, 2)
        self.assertEqual(d2_module.split_method, 'group')
        
        # Check that splits were created
        self.assertTrue(hasattr(d2_module, 'train_indices'))
        self.assertTrue(hasattr(d2_module, 'val_indices'))
        self.assertTrue(hasattr(d2_module, 'test_indices'))
        
        # Check that each split only contains windows from the assigned groups
        for idx in d2_module.train_indices:
            group_idx, _ = d2_module.mapping[idx]
            self.assertIn(group_idx, train_groups)
            
        for idx in d2_module.val_indices:
            group_idx, _ = d2_module.mapping[idx]
            self.assertIn(group_idx, val_groups)
            
        for idx in d2_module.test_indices:
            group_idx, _ = d2_module.mapping[idx]
            self.assertIn(group_idx, test_groups)
    
    def test_get_window(self):
        """Test _get_window method."""
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
            memory_efficient=False
        )
        
        # Get a window from the dataset
        if d2_module.train_indices:
            idx = d2_module.train_indices[0]
            group_idx, local_idx = d2_module.mapping[idx]
            window = d2_module._get_window(group_idx, local_idx)
            
            # Check that the window has the expected format
            self.assertTrue('past_features' in window)
            self.assertTrue('past_time' in window)
            self.assertTrue('future_targets' in window)
            self.assertTrue('future_time' in window)
            
            # Check dimensions
            self.assertEqual(window['past_features'].shape[0], d2_module.past_len)
            self.assertEqual(window['past_features'].shape[1], len(self.feature_cols))
            self.assertEqual(window['future_targets'].shape[0], d2_module.future_len)
            self.assertEqual(window['future_targets'].shape[1], len(self.target_cols))
            self.assertEqual(len(window['past_time']), d2_module.past_len)
            self.assertEqual(len(window['future_time']), d2_module.future_len)
    
    def test_known_unknown_override(self):
        """Test overriding known and unknown columns."""
        # Specify custom known and unknown columns
        known_cols = ['feature_0']
        unknown_cols = ['feature_1', 'target_0']
        
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
            memory_efficient=False,
            known_cols=known_cols,
            unknown_cols=unknown_cols
        )
        
        # Check that known and unknown columns were set correctly
        self.assertListEqual(d2_module.known_cols, known_cols)
        self.assertListEqual(d2_module.unknown_cols, unknown_cols)
        
        # Check metadata
        self.assertListEqual(d2_module.metadata['known_cols'], known_cols)
        self.assertListEqual(d2_module.metadata['unknown_cols'], unknown_cols)
        self.assertListEqual(d2_module.metadata['known_num_cols'], known_cols)  # feature_0 is numerical
        self.assertEqual(len(d2_module.metadata['known_cat_cols']), 0)  # No categorical known cols
    
    def test_getitem(self):
        """Test __getitem__ method."""
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
            memory_efficient=False
        )
        
        # Get an item from the dataset
        if d2_module.train_indices:
            idx = d2_module.train_indices[0]
            item = d2_module[idx]
            
            # Check that the item has the expected format
            self.assertTrue('past_features' in item)
            self.assertTrue('past_time' in item)
            self.assertTrue('future_targets' in item)
            self.assertTrue('future_time' in item)
            self.assertTrue('group_id' in item)
            
            # Check dimensions
            self.assertEqual(item['past_features'].shape[0], d2_module.past_len)
            self.assertEqual(item['past_features'].shape[1], len(self.feature_cols))
            self.assertEqual(item['future_targets'].shape[0], d2_module.future_len)
            self.assertEqual(item['future_targets'].shape[1], len(self.target_cols))
    
    def test_compute_valid_indices(self):
        """Test _compute_valid_indices method."""
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
            memory_efficient=False
        )
        
        # Check that valid indices were computed
        self.assertTrue(hasattr(d2_module, 'valid_indices'))
        self.assertTrue(len(d2_module.valid_indices) > 0)
        
        # Each valid index should have enough data points
        for group_idx, valid_idx_list in d2_module.valid_indices.items():
            group_data = d2_module.d1_dataset[group_idx]
            for local_idx in valid_idx_list:
                # Should have enough past and future data
                self.assertTrue(local_idx + d2_module.future_len <= len(group_data['t']))
                self.assertTrue(local_idx - d2_module.past_len >= 0)
    
    def test_create_splits(self):
        """Test _create_splits method."""
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
            memory_efficient=False
        )
        
        # Check that splits were created
        self.assertTrue(hasattr(d2_module, 'train_indices'))
        self.assertTrue(hasattr(d2_module, 'val_indices'))
        self.assertTrue(hasattr(d2_module, 'test_indices'))
        
        # Check that all indices in splits are valid
        all_indices = d2_module.train_indices + d2_module.val_indices + d2_module.test_indices
        for idx in all_indices:
            self.assertTrue(0 <= idx < len(d2_module.mapping))
    
    def test_dataloaders(self):
        """Test train/val/test dataloader methods."""
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=2,  # Small batch size for testing
            min_valid_length=4,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
            memory_efficient=False,
            num_workers=0  # Use 0 workers for testing
        )
        
        # Get dataloaders
        train_loader = d2_module.train_dataloader()
        val_loader = d2_module.val_dataloader()
        test_loader = d2_module.test_dataloader()
        
        # Check that dataloaders were created
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        
        # Test loader might be None if no test data
        if len(d2_module.test_indices) > 0:
            self.assertIsInstance(test_loader, DataLoader)
        
        # Check that we can iterate through the dataloaders
        if len(d2_module.train_indices) > 0:
            batch = next(iter(train_loader))
            self.assertTrue(isinstance(batch, dict))
            self.assertTrue('past_features' in batch)
            self.assertTrue('future_targets' in batch)
            
            # Check batch dimensions
            self.assertEqual(batch['past_features'].shape[0], min(2, len(d2_module.train_indices)))
            self.assertEqual(batch['past_features'].shape[2], len(self.feature_cols))
            self.assertEqual(batch['future_targets'].shape[0], min(2, len(d2_module.train_indices)))
            self.assertEqual(batch['future_targets'].shape[2], len(self.target_cols))


class TestTimeSeriesSubset(unittest.TestCase):
    """Test cases for the TimeSeriesSubset class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV files
        self.create_test_files()
        
        # Create D1 dataset
        self.file_paths = [
            os.path.join(self.temp_dir, f'test_data_{i}.csv') for i in range(1)
        ]
        self.group_cols = 'group'
        self.time_col = 'time'
        self.feature_cols = ['feature_0', 'feature_1']
        self.target_cols = ['target_0']
        
        self.d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            memory_efficient=False
        )
        
        # Create D2 module
        self.d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
            memory_efficient=False
        )
        
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test CSV files."""
        # Generate one CSV file
        data = []
        
        # Generate data for each group
        for group_idx in range(2):
            # Generate time series for this group
            for t in range(20):
                row = {
                    'group': f'group_{group_idx}',
                    'time': t,
                    'feature_0': np.sin(t/10 + group_idx) + np.random.normal(0, 0.1),
                    'feature_1': np.cos(t/10 + group_idx) + np.random.normal(0, 0.1),
                    'target_0': np.sin(t/5 + group_idx) + np.random.normal(0, 0.1)
                }
                data.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.temp_dir, 'test_data_0.csv'), index=False)
    
    def test_init(self):
        """Test initialization."""
        # Create a subset with the first 5 indices
        indices = list(range(5))
        subset = TimeSeriesSubset(self.d2_module, indices)
        
        # Check that the subset was initialized correctly
        self.assertEqual(subset.data_module, self.d2_module)
        self.assertEqual(subset.indices, indices)
    
    def test_len(self):
        """Test __len__ method."""
        # Create a subset with the first 5 indices
        indices = list(range(5))
        subset = TimeSeriesSubset(self.d2_module, indices)
        
        # Check that the length is correct
        self.assertEqual(len(subset), 5)
    
    def test_getitem(self):
        """Test __getitem__ method."""
        # Create a subset with all train indices
        subset = TimeSeriesSubset(self.d2_module, self.d2_module.train_indices)
        
        # Get an item from the subset
        if len(subset) > 0:
            item = subset[0]
            
            # Check that the item has the expected format
            self.assertTrue('past_features' in item)
            self.assertTrue('past_time' in item)
            self.assertTrue('future_targets' in item)
            self.assertTrue('future_time' in item)
            
            # Check dimensions
            self.assertEqual(item['past_features'].shape[0], self.d2_module.past_len)
            self.assertEqual(item['past_features'].shape[1], len(self.feature_cols))
            self.assertEqual(item['future_targets'].shape[0], self.d2_module.future_len)
            self.assertEqual(item['future_targets'].shape[1], len(self.target_cols))


class TestCustomCollateFn(unittest.TestCase):
    """Test cases for the custom_collate_fn function."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample batch data
        self.batch = [
            {
                'past_features': torch.randn(5, 2),
                'future_targets': torch.randn(2, 1),
                'past_time': [1, 2, 3, 4, 5],
                'future_time': [6, 7],
                'group_id': 'group_0',
                'st': [10.0]
            },
            {
                'past_features': torch.randn(5, 2),
                'future_targets': torch.randn(2, 1),
                'past_time': [11, 12, 13, 14, 15],
                'future_time': [16, 17],
                'group_id': 'group_1',
                'st': [20.0]
            }
        ]
    
    def test_custom_collate_fn(self):
        """Test custom_collate_fn."""
        # Apply custom collate function
        result = custom_collate_fn(self.batch)
        
        # Check that the result has the expected format
        self.assertTrue('past_features' in result)
        self.assertTrue('future_targets' in result)
        self.assertTrue('past_time' in result)
        self.assertTrue('future_time' in result)
        self.assertTrue('group_id' in result)
        self.assertTrue('st' in result)
        
        # Check that tensors were stacked correctly
        self.assertEqual(result['past_features'].shape, (2, 5, 2))
        self.assertEqual(result['future_targets'].shape, (2, 2, 1))
        
        # Check that non-tensors were kept as lists
        self.assertEqual(len(result['past_time']), 2)
        self.assertEqual(len(result['future_time']), 2)
        self.assertEqual(len(result['group_id']), 2)
        self.assertEqual(len(result['st']), 2)
        
        # Check specific values
        self.assertEqual(result['group_id'][0], 'group_0')
        self.assertEqual(result['group_id'][1], 'group_1')
        self.assertEqual(result['past_time'][0], [1, 2, 3, 4, 5])
        self.assertEqual(result['past_time'][1], [11, 12, 13, 14, 15])


if __name__ == '__main__':
    unittest.main()
