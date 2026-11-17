"""Compatibility wrapper for the imbalanced feature screener package.

The implementation lives in the src/imb_feature_screener package, but this root-level
module remains available so existing scripts continue to work while the project is
migrated to the uv-based package layout.
"""

from imb_feature_screener.feature_screener import ImbalancedFeatureScreener, main

__all__ = ["ImbalancedFeatureScreener", "main"]
