import pandas as pd
from sqlalchemy import create_engine
from configs.settings import settings
from typing import List
import logging

logger = logging.getLogger(__name__)

class TimescaleClient:
    """Enterprise Database Client for time-series telemetry."""
    def __init__(self):
        self.engine = create_engine(settings.timescale_dsn)
        logger.info(f"TimescaleDB Engine initialized at {settings.timescale_dsn.split('@')[-1]}")

    def get_historical_features(self, volume_id: str, limit: int = 2880) -> pd.DataFrame:
        """
        Fetches windowed telemetry for a specific volume directly from disk.
        limit=2880 roughly equals 48 hours of data at 1-minute intervals.
        """
        query = f"""
            SELECT * FROM telemetry_metrics 
            WHERE volume_id = '{volume_id}' 
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """
        try:
            # Execute SQL and return directly as a Pandas DataFrame
            df = pd.read_sql(query, self.engine)
            # Reverse to maintain chronological order for LSTMs
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            logger.error(f"TimescaleDB Query Failed: {e}")
            return pd.DataFrame()

    def get_topology_data(self) -> pd.DataFrame:
        """Fetch unique topology layout mappings from the database."""
        query = """
            SELECT DISTINCT volume_id, node_id, pool_id, tier, capacity_total_gb
            FROM telemetry_metrics
        """
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Failed to fetch topology from TimescaleDB: {e}")
            return pd.DataFrame()

    def get_noisy_neighbor_baselines(self) -> pd.DataFrame:
        """Fetches aggregated latency and IOPS stats per volume to fit baselines."""
        query = """
            SELECT volume_id, 
                   AVG(avg_latency_us) as lat_mean, 
                   STDDEV(avg_latency_us) as lat_std, 
                   COUNT(avg_latency_us) as lat_n,
                   AVG(total_iops) as iops_mean, 
                   STDDEV(total_iops) as iops_std,
                   COUNT(total_iops) as iops_n
            FROM telemetry_metrics
            GROUP BY volume_id
        """
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Failed to fetch baselines from TimescaleDB: {e}")
            return pd.DataFrame()

    def get_neighbors_metrics(self, volume_ids: List[str], timestamp: pd.Timestamp) -> pd.DataFrame:
        """Fetches telemetry metrics for a list of volume IDs at a specific timestamp."""
        if not volume_ids:
            return pd.DataFrame()
        # Format list for SQL IN clause
        vols_str = ", ".join([f"'{v}'" for v in volume_ids])
        query = f"""
            SELECT volume_id, avg_latency_us, total_iops 
            FROM telemetry_metrics 
            WHERE volume_id IN ({vols_str}) AND timestamp = '{timestamp}'
        """
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Failed to fetch neighbors metrics from TimescaleDB: {e}")
            return pd.DataFrame()
