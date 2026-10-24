"""Feature definitions: sigma per time slot, served from the feature store.

Design 5.3 lists sigma as the one feature whose serving-time source is the feature store
(online side, refreshed daily). Its entity is the time slot, and every slot has one row per
trading day, so the store must return the row that was valid at the asking time.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64

slot = Entity(
    name="slot",
    join_keys=["slot"],
    value_type=ValueType.STRING,
    description="Hourly time slot of the post: 09:30, 10:30, 11:30, 12:30 or 13:30 ET",
)

slot_sigma_source = FileSource(
    name="slot_sigma_source",
    path="data/slot_sigma.parquet",
    timestamp_field="event_timestamp",
)

# ttl: a Monday row is still valid on the next Monday's morning gap, but not a week later
slot_sigma = FeatureView(
    name="slot_sigma",
    entities=[slot],
    ttl=timedelta(days=4),
    schema=[Field(name="sigma", dtype=Float64)],
    source=slot_sigma_source,
    description="Std of the two-bar return at the slot over the previous 20 trading days",
)
