"""V3.1 reuses V3 fusion with a weaker configured scale-prior multiplier."""

from dqr_v3.query_fusion import MLP, ScaleAwareRegionGuidedDNFusion

__all__ = ["MLP", "ScaleAwareRegionGuidedDNFusion"]
