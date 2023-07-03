import numpy as np
from AKLTOverlap import construct_projectors,calc_eta

#===========================================================================

# kagome
# 21it [00:18,  1.13it/s] eta0  0.1 tol  0.01 eta  0.16673098088249844
# 21it [00:21,  1.00s/it] eta0  0.1 tol  0.01 eta  0.1675253903693386
# 31it [00:31,  1.03s/it] eta0  0.1 tol  0.001 eta  0.17013241128900203
# 1151it [16:41,  1.15it/s] eta0  0.1 tol  0 eta  0.17067852082351342

#spins=[4,4,4,4,4,4, 4,4,4, 4,4,4,4,4,4]
#def tri(a,b,c):
#    return [a,b,b,c,c,a]
#conns=tri(1,2,3)+tri(3,4,8)+tri(4,5,6)+tri(7,8,9)+tri(9,11,13)+tri(10,11,12)+tri(13,14,15)
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,6)
#middle=range(6,9)
#right=range(9,15)

#===========================================================================
# degree, overlap_depth, extend_depth, eta (tolerance)
# 2,2,2, 0.1285714285714294(0.0001)
# 3,2,2, 0.16203878351190582(0.0001)
# 4,2,2, 0.17514181478122304(0.0001)
# 5,2,2, 0.1814735752769116(0.0001)
# 6,2,2, 0.18509230221811623(0.0001)

degree,overlap_depth,extend_depth=6,2,2
depth=extend_depth*2+overlap_depth
degrees=[degree]*depth
conns=[]
for i in range(depth-1):
    conns+=[i,i+1]
connsL=np.array(conns).reshape(-1,2)
connsM=connsL
connsR=connsL
connsLM=connsL
connsMR=connsL
left=range(0,extend_depth)
middle=range(extend_depth,extend_depth+overlap_depth)
right=range(extend_depth+overlap_depth,2*extend_depth+overlap_depth)
    
#===========================================================================


E,F,dimLMR=construct_projectors(degrees,left,middle,right,connsL,connsM,connsR,connsLM,connsMR)

calc_eta(E,F,dimLMR,eta0=0.1,tol=0.0001)