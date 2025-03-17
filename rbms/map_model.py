from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.classes import EBM
from rbms.potts_bernoulli.classes import PBRBM

map_model: dict[str, EBM] = {"BBRBM": BBRBM, "PBRBM": PBRBM}
