"""Legacy entry point backed by the one semcs_core.Engine."""
from semcs_core import LegacyFacade
from .method import SubsetBayesMethod, ClosedSubsetBayesMethod

class Main(LegacyFacade):
    default_verbose=True
    def __init__(self, Sampler):
        closed=Sampler is not None and getattr(Sampler,"BayStc",None) is not None and Sampler.BayStc.G.get("method_profile")=="closed_ties_unit_fallback_v1"
        method=ClosedSubsetBayesMethod() if closed else SubsetBayesMethod()
        super().__init__(Sampler, method)

main = Main
