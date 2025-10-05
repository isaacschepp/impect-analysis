"""
Impect API Client Wrapper for USLC Goalkeeper Analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from impectPy import getAccessToken, Impect
import requests
from config import IMPECT_EMAIL, IMPECT_PASSWORD, GOALKEEPER_POSITIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImpectClient:
    """
    Wrapper for Impect API to fetch goalkeeper data from USLC iterations
    """
    
    def __init__(self, email: str = IMPECT_EMAIL, password: str = IMPECT_PASSWORD):
        """
        Initialize the Impect API client
        
        Args:
            email: Impect API email
            password: Impect API password
        """
        self.email = email
        self.password = password
        self.session = None
        self.token = None
        self.client = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Impect API and get access token"""
        try:
            logger.info("Authenticating with Impect API...")
            self.session = requests.Session()
            self.token = getAccessToken(self.email, self.password, self.session)
            self.client = Impect()
            logger.info("Successfully authenticated with Impect API")
        except Exception as e:
            logger.error(f"Failed to authenticate with Impect API: {e}")
            raise
    
    def get_iteration_data(self, iteration_id: int) -> pd.DataFrame:
        """
        Get all match data for a specific iteration
        
        Args:
            iteration_id: The USLC iteration ID
            
        Returns:
            DataFrame containing iteration data
        """
        try:
            logger.info(f"Fetching data for iteration {iteration_id}...")
            # Get player scores for the iteration (all positions)
            data = self.client.getPlayerIterationScores(iteration=iteration_id, positions=[])
            logger.info(f"Retrieved {len(data)} player records for iteration {iteration_id}")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch iteration data: {e}")
            raise
    
    def get_goalkeeper_data(self, iteration_id: int) -> pd.DataFrame:
        """
        Get goalkeeper-specific data for an iteration
        
        Args:
            iteration_id: The USLC iteration ID
            
        Returns:
            DataFrame containing goalkeeper data only
        """
        data = self.get_iteration_data(iteration_id)
        
        # Filter for goalkeepers based on position
        if 'position' in data.columns:
            gk_data = data[data['position'].isin(GOALKEEPER_POSITIONS)]
        elif 'positionName' in data.columns:
            gk_data = data[data['positionName'].isin(GOALKEEPER_POSITIONS)]
        else:
            # Try to identify goalkeepers by other means
            logger.warning("Position column not found, attempting alternative filtering")
            gk_data = data
        
        logger.info(f"Found {len(gk_data)} goalkeepers in iteration {iteration_id}")
        return gk_data
    
    def get_goalkeeper_match_scores(self, player_id: int, iteration_id: int) -> pd.DataFrame:
        """
        Get detailed match-by-match scores for a specific goalkeeper
        
        Args:
            player_id: The player's unique ID
            iteration_id: The iteration ID
            
        Returns:
            DataFrame containing match scores
        """
        try:
            match_scores = self.client.getPlayerMatchScores(
                player=player_id,
                iteration=iteration_id
            )
            return match_scores
        except Exception as e:
            logger.error(f"Failed to fetch match scores for player {player_id}: {e}")
            return pd.DataFrame()
    
    def get_multiple_iterations(self, iteration_ids: List[int]) -> pd.DataFrame:
        """
        Get goalkeeper data from multiple iterations and combine them
        
        Args:
            iteration_ids: List of iteration IDs
            
        Returns:
            Combined DataFrame from all iterations
        """
        all_data = []
        
        for iteration_id in iteration_ids:
            try:
                gk_data = self.get_goalkeeper_data(iteration_id)
                gk_data['iteration_id'] = iteration_id
                all_data.append(gk_data)
            except Exception as e:
                logger.error(f"Failed to process iteration {iteration_id}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined data from {len(iteration_ids)} iterations: {len(combined_data)} total records")
            return combined_data
        else:
            logger.error("No data retrieved from any iteration")
            return pd.DataFrame()
    
    def get_player_profile(self, player_id: int) -> Dict:
        """
        Get detailed profile information for a player
        
        Args:
            player_id: The player's unique ID
            
        Returns:
            Dictionary containing player profile data
        """
        try:
            profile = self.client.getPlayerProfileScores(playerId=player_id)
            return profile
        except Exception as e:
            logger.error(f"Failed to fetch profile for player {player_id}: {e}")
            return {}
