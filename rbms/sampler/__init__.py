from rbms.classes import Sampler
from rbms.sampler.cd import CD
from rbms.sampler.pcd import PCD
from rbms.sampler.rdm import RDM

map_sampler: dict[str, Sampler] = {"PCD": PCD, "CD": CD, "RDM": RDM}
