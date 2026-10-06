from .base import Registry
from .engines.biological import MetabolicNetworkEngine
from .engines.biophysical import EnzymeCycleEngine
from .engines.electronic import EVBFlexibleProtonTransferEngine, EVBProtonTransferEngine
from .engines.molecular import StationaryPointEngine
from .engines.qc_scan import QCScanEngine
from .engines.quantum import DoubleWellEngine, TabulatedPotentialEngine
from .engines.reaction import MassActionEngine
from .translators.biophysical_to_pathway import BiophysicalToPathway
from .translators.electronic_to_molecular import ElectronicToMolecular
from .translators.electronic_to_quantum import ElectronicToQuantum
from .translators.molecular_to_reaction import MolecularToReaction
from .translators.quantum_to_reaction import QuantumToReaction
from .translators.reaction_to_enzyme import ReactionToEnzyme


def default_registry() -> Registry:
    registry = Registry()
    engines = (
        EVBProtonTransferEngine(), EVBFlexibleProtonTransferEngine(), StationaryPointEngine(),
        DoubleWellEngine(), TabulatedPotentialEngine(), MassActionEngine(), EnzymeCycleEngine(),
        MetabolicNetworkEngine(), QCScanEngine(),
    )
    for engine in engines:
        registry.register_engine(engine)
    for translator in (ElectronicToQuantum(), ElectronicToMolecular(), QuantumToReaction(), MolecularToReaction(), ReactionToEnzyme(),
                       BiophysicalToPathway()):
        registry.register_translator(translator)
    return registry
