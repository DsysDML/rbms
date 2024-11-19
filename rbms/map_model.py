from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.classes import RBM
from rbms.potts_bernoulli.classes import PBRBM

map_model: dict[str, RBM] = {"BBRBM": BBRBM, "PBRBM": PBRBM}
