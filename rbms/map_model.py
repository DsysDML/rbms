from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.classes import EBM
from rbms.potts_bernoulli.classes import PBRBM
from rbms.ising_ising.classes import IIRBM
from rbms.bernoulli_gaussian.classes import BGRBM

map_model: dict[str, EBM] = {
    "BBRBM": BBRBM,
    "PBRBM": PBRBM,
    "IIRBM": IIRBM,
    "BGRBM": BGRBM,
}
