import numpy

# ................................................
# generate random a, b and c for each parent in mi size population from normal distrib, -10<=a,b,c<=10
# generate random sa, sb and sc for each parent in mi size population from normal distrib, 0<=sa,sb,sc<=10


# A = [[1, 2, 3], [4, 5, 6]]
# A = numpy.append(A, [[2, 2, 2]], axis=0)
# ................................................

n = 6
tau1 = 1/numpy.sqrt(2*n)
tau2 = 1/numpy.sqrt(2*numpy.sqrt(n))
mi_size = 1000
lambda_size = mi_size*5
mi = [['a', 'b', 'c', 'sa', 'sb', 'sc']]
# mi = numpy.append(mi, [['aa', 'bb', 'cc', 'saa', 'sbb', 'scc']], axis=0)
print(mi)
print(mi[0][5])

for i in range(mi_size):  # make initial population
    a = 1  # change to normal dist val from -10 to 10
    sa = 11
    b = 2  # change to normal dist val from -10 to 10
    sb = 22
    c = 3  # change to normal dist val from -10 to 10
    sc = 33
    if i==0:
        mi = [[a+1, b+1, c+1, sa+1, sb+1, sc+1]]
    else:
        mi = numpy.append(mi, [[a, b, c, sa, sb, sc]], axis=0)

for i in range(mi_size): # make mutants XDD
    # lambda = 5mi, variable name lambda_pop since "lambda" itself is reserved
    # -> each parent should be mutated 5 times
    for j in range(5):  # 0 = a, 1 = b, 2 = c, 3 = sa, 4 = sb, 5 = sc
        r1 = tau1 * numpy.random.normal(0, 1)
        a_offspring = mi[i][0] + mi[i][3] * numpy.random.normal(0, 1)
        sa_offspring = mi[i][3] * numpy.exp(r1) * numpy.exp(tau2 * numpy.random.normal(0, 1))
        b_offspring = mi[i][1] + mi[i][4] * numpy.random.normal(0, 1)
        sb_offspring = mi[i][4] * numpy.exp(r1) * numpy.exp(tau2 * numpy.random.normal(0, 1))
        c_offspring = mi[i][2] + mi[i][5] * numpy.random.normal(0, 1)
        sc_offspring = mi[i][5] * numpy.exp(r1) * numpy.exp(tau2 * numpy.random.normal(0, 1))
        if i == 0:
            lambda_pop = [[a_offspring, b_offspring, c_offspring, sa_offspring, sb_offspring, sc_offspring]]
        else:
            lambda_pop = numpy.append(lambda_pop,
                                      [[a_offspring, b_offspring, c_offspring, sa_offspring, sb_offspring, sc_offspring]], axis=0)

print(mi)

print(lambda_pop)
# todo:
#  import data from file (?) data8.dat
#  add function to calc MSE between o and o_hat
#  change the range of initial val of abc and sa sb sc (HOW?!)
#  calculate fitness val for every new individual by combining mi and lambda_pop, pick the new mi
#  change the t = t+1, if t<N_max -> again generate mutants etc. (loop; Nmax = 200)
#  change the range of initial val of abc,
