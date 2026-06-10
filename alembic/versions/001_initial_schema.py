"""initial_schema

Revision ID: 001
Revises: 
Create Date: 2026-06-10 21:56:26

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_metrics (
            timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            volume_id VARCHAR(50) NOT NULL,
            iops FLOAT,
            latency_ms FLOAT,
            bandwidth_mbps FLOAT,
            capacity_used_pct FLOAT,
            PRIMARY KEY (volume_id, timestamp)
        );
        SELECT create_hypertable('telemetry_metrics', 'timestamp', if_not_exists => TRUE);
    """)

def downgrade():
    op.execute("DROP TABLE IF EXISTS telemetry_metrics;")
