"""
Data Collector for USLC Goalkeeper Analysis
Fetches and caches data from Impect API
"""

import pandas as pd
import os
import pickle
from typing import List, Dict
import logging
from impect_client import ImpectClient
from config import TRAINING_ITERATIONS, DATA_CACHE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoalkeeperDataCollector:
    """
    Collects and manages goalkeeper data from USLC iterations
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the data collector
        
        Args:
            use_cache: Whether to use cached data if available
        """
        self.use_cache = use_cache
        self.client = None
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the path for a cache file"""
        return os.path.join(DATA_CACHE_DIR, f"{cache_key}.pkl")
    
    def _load_from_cache(self, cache_key: str) -> pd.DataFrame:
        """Load data from cache"""
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            logger.info(f"Loading data from cache: {cache_key}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """Save data to cache"""
        cache_path = self._get_cache_path(cache_key)
        logger.info(f"Saving data to cache: {cache_key}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def collect_training_data(self) -> pd.DataFrame:
        """
        Collect all training data from configured USLC iterations
        
        Returns:
            Combined DataFrame with all goalkeeper data
        """
        cache_key = "training_data_all_iterations"
        
        # Try to load from cache first
        if self.use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Initialize client and fetch data
        logger.info("Fetching training data from Impect API...")
        self.client = ImpectClient()
        
        iteration_ids = list(TRAINING_ITERATIONS.values())
        all_data = self.client.get_multiple_iterations(iteration_ids)
        
        # Add year information
        year_map = {v: k for k, v in TRAINING_ITERATIONS.items()}
        all_data['year'] = all_data['iteration_id'].map(year_map)
        
        # Save to cache
        if not all_data.empty:
            self._save_to_cache(all_data, cache_key)
        
        return all_data
    
    def collect_iteration_data(self, year: int) -> pd.DataFrame:
        """
        Collect data for a specific year
        
        Args:
            year: The year (e.g., 2025, 2024)
            
        Returns:
            DataFrame with goalkeeper data for that year
        """
        if year not in TRAINING_ITERATIONS:
            raise ValueError(f"Year {year} not in configured training iterations")
        
        iteration_id = TRAINING_ITERATIONS[year]
        cache_key = f"iteration_{iteration_id}_{year}"
        
        # Try cache first
        if self.use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Fetch from API
        if self.client is None:
            self.client = ImpectClient()
        
        data = self.client.get_goalkeeper_data(iteration_id)
        data['year'] = year
        data['iteration_id'] = iteration_id
        
        # Save to cache
        if not data.empty:
            self._save_to_cache(data, cache_key)
        
        return data
    
    def get_player_history(self, player_id: int, iterations: List[int] = None) -> pd.DataFrame:
        """
        Get historical data for a specific player across iterations
        
        Args:
            player_id: The player's unique ID
            iterations: List of iteration IDs to search (defaults to training iterations)
            
        Returns:
            DataFrame with player's historical data
        """
        if iterations is None:
            iterations = list(TRAINING_ITERATIONS.values())
        
        if self.client is None:
            self.client = ImpectClient()
        
        history = []
        for iteration_id in iterations:
            try:
                data = self.client.get_goalkeeper_data(iteration_id)
                player_data = data[data['playerId'] == player_id]
                if not player_data.empty:
                    history.append(player_data)
            except Exception as e:
                logger.warning(f"Could not fetch data for iteration {iteration_id}: {e}")
                continue
        
        if history:
            return pd.concat(history, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def export_to_csv(self, data: pd.DataFrame, filename: str):
        """
        Export data to CSV file
        
        Args:
            data: DataFrame to export
            filename: Output filename
        """
        output_path = os.path.join('output', filename)
        os.makedirs('output', exist_ok=True)
        data.to_csv(output_path, index=False)
        logger.info(f"Data exported to {output_path}")
