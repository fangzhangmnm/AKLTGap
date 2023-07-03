#===========================================================================
# rectangular overlap on square spin-2 lattice
# width
# LLLLL extend_depth
# MMMMM overlap_depth
# MMMMM 
# RRRRR

# width, overlap_depth, extend_depth, eta (tolerance)

# change overlap_depth
# 2,1,1,0.638574167746431(0.0001)
# 2,2,1,0.2811252854883967(0.0001)
# 2,3,1,0.13637374015358744(0.0001)
# 2,4,1,0.06536558761045402(0.0001)

# change extend_depth
# 1,2,1,0.17647058823529538(0.0001)
# 1,2,2,0.1751418147812207(0.0001)
# 1,2,3,0.1743895601498659(0.0001)
# 1,2,4,0.17435446170372704(0.0001)

# change extend_depth
# 2,2,1,0.2811252854883967(0.0001)
# 2,2,2,0.2667441652593447(0.0001)
# 2,2,3,0.25593152877229053(0.01)21it [05:26, 15.53s/it]

# change width
# 1,2,1,0.17647058823529538(0.0001)
# 2,2,1,0.2811252854883967(0.0001)
# 3,2,1,0.3247779907844131(0.0001)
# 4,2,1,0.34254150021112584,(0.01)

width,overlap_depth,extend_depth=1,2,4

depth=2*extend_depth+overlap_depth
degrees=[4]*(width*depth)
conns=[]
for i in range(depth):
    for j in range(width-1):
        conns+=[i*width+j,i*width+j+1]

for i in range(depth-1):
    for j in range(width):
        conns+=[i*width+j,(i+1)*width+j]
print(spins,conns)
connsL=np.array(conns).reshape(-1,2)
connsM=connsL
connsR=connsL
connsLM=connsL
connsMR=connsL
left=range(0,extend_depth*width)
middle=range(extend_depth*width,(extend_depth+overlap_depth)*width)
right=range((extend_depth+overlap_depth)*width,(2*extend_depth+overlap_depth)*width)
#===========================================================================
# spin chain with different spins (spin=degree//2)
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