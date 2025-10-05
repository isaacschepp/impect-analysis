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
    
    def get_matches(self, iteration_id: int) -> pd.DataFrame:
        """
        Get all matches for a specific iteration with results
        
        Args:
            iteration_id: The USLC iteration ID
            
        Returns:
            DataFrame containing match data with results (wins/draws/losses)
        """
        try:
            logger.info(f"Fetching matches for iteration {iteration_id}...")
            matches = self.client.getMatches(iteration=iteration_id)
            logger.info(f"Retrieved {len(matches)} matches for iteration {iteration_id}")
            return matches
        except Exception as e:
            logger.error(f"Failed to fetch matches: {e}")
            return pd.DataFrame()
    
    def get_player_match_level_data(self, iteration_id: int, positions: List[str] = None) -> pd.DataFrame:
        """
        Get player-level data for all matches in an iteration
        This includes match-by-match performance
        
        Args:
            iteration_id: The USLC iteration ID
            positions: List of positions to filter (default: GOALKEEPER_POSITIONS)
            
        Returns:
            DataFrame with player match data
        """
        if positions is None:
            positions = GOALKEEPER_POSITIONS
            
        try:
            # First get all matches
            matches = self.get_matches(iteration_id)
            if matches.empty:
                return pd.DataFrame()
            
            # Get match IDs
            match_ids = matches['matchId'].tolist() if 'matchId' in matches.columns else []
            
            if not match_ids:
                logger.warning(f"No match IDs found for iteration {iteration_id}")
                return pd.DataFrame()
            
            logger.info(f"Fetching player data for {len(match_ids)} matches...")
            
            # Get player scores for these matches
            player_match_data = self.client.getPlayerMatchScores(
                matches=match_ids,
                positions=positions
            )
            
            logger.info(f"Retrieved player match data: {len(player_match_data)} records")
            
            # Merge with match results to get wins/draws
            if not matches.empty and not player_match_data.empty:
                player_match_data = player_match_data.merge(
                    matches[['matchId', 'homeTeamId', 'awayTeamId', 'homeGoals', 'awayGoals']],
                    on='matchId',
                    how='left'
                )
            
            return player_match_data
            
        except Exception as e:
            logger.error(f"Failed to fetch player match level data: {e}")
            return pd.DataFrame()
    
    def get_comprehensive_goalkeeper_data(self, iteration_id: int) -> pd.DataFrame:
        """
        Get comprehensive goalkeeper data including match results for win/draw/loss tracking
        
        This method:
        1. Fetches iteration-level aggregated stats (getPlayerIterationScores)
        2. Fetches match-level data with results (getPlayerMatchScores + getMatches)
        3. Calculates points gained per match (3 for win, 1 for draw, 0 for loss)
        4. Aggregates match results to player level
        
        Args:
            iteration_id: The USLC iteration ID
            
        Returns:
            DataFrame with comprehensive goalkeeper data including match results
        """
        try:
            # Get basic iteration stats
            logger.info(f"Fetching comprehensive data for iteration {iteration_id}...")
            iteration_data = self.get_goalkeeper_data(iteration_id)
            
            if iteration_data.empty:
                logger.warning(f"No goalkeeper data found for iteration {iteration_id}")
                return pd.DataFrame()
            
            # Get match-level data with results
            match_data = self.get_player_match_level_data(iteration_id, GOALKEEPER_POSITIONS)
            
            if match_data.empty:
                logger.warning(f"No match-level data found for iteration {iteration_id}")
                return iteration_data
            
            # Calculate match results for each player
            match_results = self._calculate_match_results(match_data)
            
            # Merge match results with iteration data
            if 'playerId' in iteration_data.columns and not match_results.empty:
                comprehensive_data = iteration_data.merge(
                    match_results,
                    on='playerId',
                    how='left'
                )
                
                # Fill NaN values for goalkeepers without match result data
                result_cols = ['matches_played_with_result', 'wins', 'draws', 'losses', 
                             'points_gained', 'points_per_match']
                for col in result_cols:
                    if col in comprehensive_data.columns:
                        comprehensive_data[col] = comprehensive_data[col].fillna(0)
                
                logger.info(f"Comprehensive data prepared with {len(comprehensive_data)} goalkeepers")
                return comprehensive_data
            else:
                logger.warning("Could not merge match results with iteration data")
                return iteration_data
                
        except Exception as e:
            logger.error(f"Failed to get comprehensive goalkeeper data: {e}")
            return pd.DataFrame()
    
    def _calculate_match_results(self, match_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate match results (wins/draws/losses) and points from match-level data
        
        Args:
            match_data: DataFrame with player match data including team IDs and goals
            
        Returns:
            DataFrame with aggregated match results per player
        """
        if match_data.empty:
            return pd.DataFrame()
        
        # Determine if player's team won, drew, or lost
        def get_result(row):
            try:
                team_id = row.get('teamId', None)
                home_id = row.get('homeTeamId', None)
                away_id = row.get('awayTeamId', None)
                home_goals = row.get('homeGoals', 0)
                away_goals = row.get('awayGoals', 0)
                
                if pd.isna(team_id) or pd.isna(home_goals) or pd.isna(away_goals):
                    return None, 0
                
                # Determine if player was on home or away team
                is_home = (team_id == home_id)
                
                # Calculate result
                if home_goals > away_goals:
                    # Home team won
                    result = 'win' if is_home else 'loss'
                    points = 3 if is_home else 0
                elif home_goals < away_goals:
                    # Away team won
                    result = 'loss' if is_home else 'win'
                    points = 0 if is_home else 3
                else:
                    # Draw
                    result = 'draw'
                    points = 1
                
                return result, points
            except Exception as e:
                logger.debug(f"Error calculating result for row: {e}")
                return None, 0
        
        # Apply result calculation
        match_data[['result', 'points']] = match_data.apply(
            lambda row: pd.Series(get_result(row)), axis=1
        )
        
        # Aggregate by player
        player_results = match_data.groupby('playerId').agg({
            'matchId': 'count',  # matches played
            'result': lambda x: (x == 'win').sum(),  # wins
            'points': 'sum'  # total points
        }).reset_index()
        
        player_results.columns = ['playerId', 'matches_played_with_result', 'wins', 'points_gained']
        
        # Calculate draws and losses
        match_data_grouped = match_data.groupby('playerId')['result'].value_counts().unstack(fill_value=0)
        
        if 'draw' in match_data_grouped.columns:
            player_results = player_results.merge(
                match_data_grouped[['draw']].reset_index().rename(columns={'draw': 'draws'}),
                on='playerId',
                how='left'
            )
        else:
            player_results['draws'] = 0
        
        if 'loss' in match_data_grouped.columns:
            player_results = player_results.merge(
                match_data_grouped[['loss']].reset_index().rename(columns={'loss': 'losses'}),
                on='playerId',
                how='left'
            )
        else:
            player_results['losses'] = 0
        
        # Calculate points per match
        player_results['points_per_match'] = (
            player_results['points_gained'] / player_results['matches_played_with_result']
        ).fillna(0)
        
        return player_results
