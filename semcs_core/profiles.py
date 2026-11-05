"""Generic validation/merge of a model-owned sampling profile specification.

New method classes supply sampler_profile_spec(profile); no core method-family
registration or algorithm-specific default is required.
"""
import copy
from collections.abc import Mapping

def configure_method_profile(model, family, method_profile=None):
    if method_profile is not None and (not isinstance(method_profile,str) or not method_profile):
        raise ValueError("Method profile name must be a nonempty string")
    provider=getattr(model,"sampler_profile_spec",None)
    if not callable(provider):raise TypeError("Model must supply sampler_profile_spec(profile)")
    spec=provider() if method_profile is None else provider(method_profile)
    if not isinstance(spec,Mapping) or spec.get("family")!=family:
        raise ValueError("Model profile family does not match the requested family")
    profile_name=spec.get("name")
    if not isinstance(profile_name,str) or not profile_name:raise ValueError("Model profile must supply its canonical name")
    kernels=spec.get("kernels",{});evidence=spec.get("evidence",{});method_state=spec.get("state",{})
    if not isinstance(kernels,Mapping) or not isinstance(evidence,Mapping):
        raise TypeError("Model profile kernels/evidence must be mappings")
    if not isinstance(method_state,Mapping):raise TypeError("Model profile state must be a mapping")
    from .MCMCBase import MCMCBase
    g=copy.deepcopy(model.G);settings=copy.deepcopy(g.get("sampler",{}))
    names=list(settings)
    for name in ("aCS","ES","CW_aMH","HMC",*kernels):
        if name not in names:names.append(name)
    for name in names:
        override=kernels.get(name,{})
        if not isinstance(name,str) or not isinstance(override,Mapping):
            raise TypeError("Kernel profile overrides must be named mappings")
        settings[name]={**MCMCBase._defaults,**settings.get(name,{}),**copy.deepcopy(override)}
    g.update(method_profile=profile_name,method_family=family,sampler=settings)
    g.update(copy.deepcopy(dict(method_state)))
    g["profile_evidence"]=copy.deepcopy(dict(evidence))
    model.G=g
    return model
