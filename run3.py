import numpy as np
from AKLTOverlap import construct_projectors,calc_eta



E,F,dimLMR=construct_projectors(spins,left,middle,right,connsL,connsM,connsR,connsLM,connsMR)

calc_eta(E,F,dimLMR,eta0=0.1,tol=0.01)