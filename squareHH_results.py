#===========================================================================
# square HH diag
#spins=[4]*28
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(1,2,3,4)+quad(5,6,7,8)+quad(9,10,11,12)+quad(13,14,15,16)+quad(17,18,19,20)+quad(21,22,23,24)+quad(25,26,27,28)\
#        +quad(4,7,10,13)+quad(16,19,22,25)\
#        +[3,9,8,14,15,21,20,26]
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,12)
#middle=range(12,16)
#right=range(16,28)
#
#===========================================================================
# square HH side
#spins=[4]*24
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#connsLM0=quad(1,2,3,4)+quad(5,6,7,8)+quad(9,10,11,12)+quad(13,14,15,16)+quad(4,11,6,13)+[3,5,12,14]
#connsMR0=quad(9,10,11,12)+quad(13,14,15,16)+quad(17,18,19,20)+quad(21,22,23,24)+quad(12,19,14,21)+[10,17,16,23]
#connsL=np.array(connsLM0).reshape(-1,2)-1
#connsM=np.array(connsMR0).reshape(-1,2)-1
#connsR=np.array(connsMR0).reshape(-1,2)-1
#connsLM=np.array(connsLM0).reshape(-1,2)-1
#connsMR=np.array(connsMR0).reshape(-1,2)-1
#left=range(0,8)
#middle=range(8,16)
#right=range(16,24)
#===========================================================================
# square HH smaller1 diag
# 21it [02:24,  6.87s/it] eta0  0.1 tol  0.01 eta  0.27006527915426287
# 51it [05:57,  7.00s/it] eta0  0.1 tol  0.001 eta  0.27617431527586245

#spins=[4]*14
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(1,2,4,6)+quad(6,7,8,9)+quad(9,11,13,14)+[2,3,3,7,4,5,10,11,12,8,12,13]
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,5)
#middle=range(5,9)
#right=range(9,14)
#===========================================================================
# square HH smaller2 side
#spins=[4]*16
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#connsLM0=quad(5,6,7,8)+quad(9,10,11,12)+quad(2,7,3,9)+[1,2,3,4,8,10]
#connsMR0=quad(5,6,7,8)+quad(9,10,11,12)+quad(8,14,10,15)+[6,13,13,14,16,15,16,12]
#connsL=np.array(connsLM0).reshape(-1,2)-1
#connsM=np.array(connsMR0).reshape(-1,2)-1
#connsR=np.array(connsMR0).reshape(-1,2)-1
#connsLM=np.array(connsLM0).reshape(-1,2)-1
#connsMR=np.array(connsMR0).reshape(-1,2)-1
#left=range(0,4)
#middle=range(4,12)
#right=range(12,16)
#===========================================================================
# square HH smaller3 diag
# 21it [2:35:23, 443.96s/it] eta0  0.1 tol  0.01 eta  0.27013316288219225
#spins=[4]*20
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(5,6,7,8)+quad(2,3,6,9)+quad(9,10,11,12)+quad(12,15,18,19)+quad(13,14,15,16)+[1,2,1,5,4,3,4,10,17,11,17,18,20,16,20,19]
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,8)
#middle=range(8,12)
#right=range(12,20)
#===========================================================================
# square HH smaller4 diag
# 21it [00:21,  1.03s/it] eta0  0.1 tol  0.01 eta  0.2710124114752804
# 51it [00:45,  1.12it/s] eta0  0.1 tol  0.001 eta  0.27657318349712146
# 1681it [25:17,  1.11it/s] eta0  0.1 tol  0.0 eta  0.2768708234480815

#spins=[4]*12
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(1,2,4,5)+quad(5,6,7,8)+quad(8,9,11,12)+[3,2,3,6,10,7,10,11]
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,4)
#middle=range(4,8)
#right=range(8,12)
#===========================================================================
# square HH smaller5 diag
# 21it [04:17, 12.26s/it] eta0  0.1 tol  0.01 eta  0.2707052746632339

#spins=[4]*16
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(1,2,4,5)+quad(3,4,6,7)+quad(7,8,9,10)+quad(10,11,13,14)+quad(12,13,15,16)+[5,8,9,12]
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,6)
#middle=range(6,10)
#right=range(10,16)
#===========================================================================
# square HH smaller6 diag
# 21it [36:25, 104.05s/it] eta0  0.1 tol  0.01 eta  0.2707780804218023

#spins=[4]*18
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(2,3,5,6)+quad(4,5,7,8)+quad(8,9,10,11)+quad(11,12,14,15)+quad(13,14,16,17)+[1,4,6,9,10,13,15,18]
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,7)
#middle=range(7,11)
#right=range(11,18)
#===========================================================================
# square HH smaller7 diag
# 21it [43:51, 125.32s/it] eta0  0.1 tol  0.01 eta  0.26623556902963275

#spins=[4]*18
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(1,2,4,5)+quad(3,4,6,8)+quad(8,9,10,11)+quad(11,13,15,16)+quad(14,15,17,18)+[5,9,6,7,12,13,10,14]
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,7)
#middle=range(7,11)
#right=range(11,18)
#===========================================================================
# square HH smaller8 diag
# 1it [3:47:26, 13646.82s/it] _arpack.error: (shape(v,0)==ldv) failed for 3rd keyword ldv: dsaupd:ldv=-2118184960
#spins=[4]*20
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(2,3,5,6)+quad(4,5,7,9)+quad(9,10,11,12)+quad(12,14,16,17)+quad(15,16,18,19)+[1,4,6,10,7,8,13,14,11,15,17,20]
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,8)
#middle=range(8,12)
#right=range(12,20)

#===========================================================================
# square HH smaller9 diag
# 21it [00:11,  1.90it/s] eta0  0.1 tol  0.01 eta  0.2674702682529123
# 41it [00:15,  2.62it/s] eta0  0.1 tol  0.001 eta  0.2710067145091738
# 1055it [07:36,  2.31it/s] eta0  0.1 tol  0.0 eta  0.2720337069909235

#spins=[4]*10
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(1,2,4,5)+quad(6,7,9,10)+[3,4,4,6,5,7,7,8]
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,3)
#middle=range(3,7)
#right=range(7,10)
#===========================================================================
# square HH smaller10 diag
# 21it [00:18,  1.14it/s] eta0  0.1 tol  0.01 eta  0.27075031085511814

#spins=[4]*10
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(4,5,6,7)+[1,4,2,5,3,4,6,9,7,10,7,8]
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,3)
#middle=range(3,7)
#right=range(7,10)
#===========================================================================
# square HH smaller11 diag
# 21it [00:04,  4.95it/s] eta0  0.1 tol  0.01 eta  0.12368761190885075

#spins=[4]*10
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(3,4,5,6)+[1,3,2,3,6,7,6,8]
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,2)
#middle=range(2,6)
#right=range(6,8)
#===========================================================================
# square HH small12 diag
# 21it [00:06,  3.23it/s] eta0  0.1 tol  0.01 eta  0.12243187694378044

#spins=[4]*10
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#conns=quad(1,2,3,4)+quad(4,5,6,7)+quad(7,8,9,10)
#connsL=np.array(conns).reshape(-1,2)-1
#connsM=connsL
#connsR=connsL
#connsLM=connsL
#connsMR=connsL
#left=range(0,3)
#middle=range(3,7)
#right=range(7,10)
#===========================================================================
# square HH smaller13 side
# 21it [02:12,  6.29s/it] eta0  0.1 tol  0.01 eta  0.32935050556181994

#spins=[4]*12
#def quad(a,b,c,d):
#    return [a,b,a,c,b,d,c,d]
#connsLM0=quad(1,5,2,7)+quad(3,4,5,6)+quad(7,8,9,10)
#connsMR0=quad(6,11,8,12)+quad(3,4,5,6)+quad(7,8,9,10)
#connsM0=quad(3,4,5,6)+quad(7,8,9,10)
#connsL=np.array(connsLM0).reshape(-1,2)-1
#connsM=np.array(connsM0).reshape(-1,2)-1
#connsR=np.array(connsMR0).reshape(-1,2)-1
#connsLM=np.array(connsLM0).reshape(-1,2)-1
#connsMR=np.array(connsMR0).reshape(-1,2)-1
#left=range(0,2)
#middle=range(2,10)
#right=range(10,12)
#===========================================================================