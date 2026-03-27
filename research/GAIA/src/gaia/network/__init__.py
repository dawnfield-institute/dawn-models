"""GAIA Network — recursive multi-agent architecture.

Same physics at every scale: modules within a bus, agents within a network.
CoupledFieldsBus is the universal coupling mechanism at all levels.

    Level 0: GAIAModule (Safety, Reasoning, Memory, ...)
    Level 1: RecursiveEntity (CoupledFieldsBus + modules, wrapped as GAIAModule)
    Level 2: GAIAAgent (RecursiveEntity + identity + self-modification)
    Level 3: GAIANetwork (CoupledFieldsBus of GAIAAgents)
"""

from .agent import GAIAAgent
from .checkpoint import save_colony, load_colony, checkpoint_info
from .identity import AgentIdentity
from .network import GAIANetwork
from .recursive_entity import RecursiveEntity

__all__ = [
    "AgentIdentity",
    "GAIAAgent",
    "GAIANetwork",
    "RecursiveEntity",
    "save_colony",
    "load_colony",
    "checkpoint_info",
]
