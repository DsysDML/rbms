from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.bernoulli_gaussian.classes import BGRBM
from rbms.bm.potts import PBM
from rbms.classes import EBM
from rbms.gaussian_bernoulli.classes import GBRBM
from rbms.gaussian_bernoulli_var.classes import GBRBM_VAR
from rbms.ising_gaussian.classes import IGRBM
from rbms.ising_ising.classes import IIRBM
from rbms.potts_bernoulli.classes import PBRBM
from rbms.truncated_gaussian_bernoulli.classes import TGBRBM

map_model: dict[str, type[EBM]] = {
    "BBRBM": BBRBM,
    "PBRBM": PBRBM,
    "BGRBM": BGRBM,
    "GBRBM": GBRBM,
    "GBRBM_VAR": GBRBM_VAR,
    "TGBRBM": TGBRBM,
    "IGRBM": IGRBM,
    "IIRBM": IIRBM,
    "PBM": PBM,
}
