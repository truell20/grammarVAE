import zinc_grammar
from models.model_zinc import MoleculeVAE
MoleculeVAE().create(zinc_grammar.GCFG.productions(), None)