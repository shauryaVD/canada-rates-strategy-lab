"""Official Canadian benchmark bond yield ingestion."""

from .bank_of_canada import DataBundle, load_data
from .validation import SERIES, TENORS, DataUnavailable, parse_observations

__all__ = ["DataBundle", "DataUnavailable", "SERIES", "TENORS", "load_data", "parse_observations"]
