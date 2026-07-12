# SPDX-License-Identifier: MIT
"""One multi-lens analyzer over corpus knowledge and witness geometry.

Increment 1 exports the unified typed complex, two adapters, and four lenses:
topology, graph, spatial, and statistics.  Vector, set, temporal, relational,
composition, and the query DSL remain explicitly outside this increment.
"""

from .adapters import CorpusAdapter, WitnessAdapter
from .core import (
    AdapterError,
    ComplexEdge,
    ComplexElement,
    ComplexValidationError,
    LensEngineError,
    LensOperationError,
    QueryError,
    SpatialGeometry,
    T,
    TimeInterval,
    TypedAttributedComplex,
    TypedRelation,
    TypedResult,
)
from .graph import (
    Centrality,
    CommunityPartition,
    ComponentPartition,
    GraphLens,
    ShortestPath,
    Traversal,
)
from .protocols import ComplexAdapter, Lens
from .query import GRAPH, LENSES, SPATIAL, STATISTICS, TOPOLOGY, query
from .spatial import (
    Distance,
    LaguerreCells,
    LaguerreDiagram,
    Overlap,
    PointContainment,
    SpatialLens,
)
from .statistics import (
    ChangePoint,
    DistributionDrift,
    KDEDensity,
    StatisticsLens,
    StructureTensorAnisotropy,
)
from .topology import (
    Basin,
    CriticalPoint,
    IntegralRoute,
    PersistencePair,
    TopologyLens,
    Watershed,
)

__all__ = [
    "GRAPH",
    "LENSES",
    "SPATIAL",
    "STATISTICS",
    "TOPOLOGY",
    "AdapterError",
    "Basin",
    "Centrality",
    "ChangePoint",
    "CommunityPartition",
    "ComplexAdapter",
    "ComplexEdge",
    "ComplexElement",
    "ComplexValidationError",
    "ComponentPartition",
    "CorpusAdapter",
    "CriticalPoint",
    "Distance",
    "DistributionDrift",
    "GraphLens",
    "IntegralRoute",
    "KDEDensity",
    "LaguerreCells",
    "LaguerreDiagram",
    "Lens",
    "LensEngineError",
    "LensOperationError",
    "Overlap",
    "PersistencePair",
    "PointContainment",
    "QueryError",
    "ShortestPath",
    "SpatialGeometry",
    "SpatialLens",
    "StatisticsLens",
    "StructureTensorAnisotropy",
    "T",
    "TimeInterval",
    "TopologyLens",
    "Traversal",
    "TypedAttributedComplex",
    "TypedRelation",
    "TypedResult",
    "Watershed",
    "WitnessAdapter",
    "query",
]
