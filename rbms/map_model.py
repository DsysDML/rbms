from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.bernoulli_gaussian.classes import BGRBM
from rbms.classes import EBM
from rbms.ising_gaussian.classes import IGRBM
from rbms.ising_ising.classes import IIRBM
from rbms.potts_bernoulli.classes import PBRBM
from rbms.EBM_binary.classes import BEBM
from rbms.EBM_continuous.classes import CEBM

map_model: dict[str, type[EBM]] = {
    "BBRBM": BBRBM,
    "PBRBM": PBRBM,
    "BGRBM": BGRBM,
    "IGRBM": IGRBM,
    "IIRBM": IIRBM,
    "BEBM": BEBM,
    "CEBM": CEBM,
}
