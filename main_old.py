import math
import numpy
import matplotlib.pyplot as plt

f = open("ES_data_8.dat")
data = numpy.empty((0, 2), float)
for line in f.read().splitlines(False):
    data = numpy.append(data, numpy.array([list(map(float, (line.lstrip().split())))]), axis=0)

f.close()

t = 0
t_best = 0
d = pow(10, -5)
n = 6
N = 101
N_max = 100
tau1 = 1/math.sqrt(2*n)
tau2 = 1/math.sqrt(2*math.sqrt(n))
mi_size = 1000
lambda_size = mi_size * 5
best_abcE = None


def mse_calc(a, b, c):
    E = 0
    for row in data:
        o_hat = a * ((row[0])**2 - b*math.cos(c * numpy.pi * row[0]))  # row[0] == i
        E = E + (row[1] - o_hat)**2

    # print("mse ->", E/N)
    return E/N


def parent_offspring_error():  # |E_o - E_p| <= d
    for i in range(mi_size):
        if mi[i][6] > 0.25:
            continue
        for j in range(6):
            if abs(mi[i][6] - lambda_pop[i + j][6]) <= d:  # and mi[i][6] <= 0.25 and lambda_pop[i + j][6] <= 0.25:
                return i+j
    return -1


mi = numpy.empty((0, 7), float)
for i in range(mi_size):  # make initial population: abc - unif dist val from -10 to 10
    a = numpy.random.uniform(-10, 10)
    sa = numpy.random.uniform(0, 10)
    b = numpy.random.uniform(-10, 10)
    sb = numpy.random.uniform(0, 10)
    c = numpy.random.uniform(-10, 10)
    sc = numpy.random.uniform(0, 10)
    mi = numpy.append(mi, [[a, b, c, sa, sb, sc, mse_calc(a, b, c)]], axis=0)

for it in range(N_max):
    print("generation: ", t)
    lambda_pop = numpy.empty((0, 7), float)
    for i in range(mi_size):  # make mutants XDD
        # lambda = 5mi, variable name lambda_pop since "lambda" itself is reserved
        # -> each parent should be mutated 5 times
        for j in range(5):  # mi args: 0 = a, 1 = b, 2 = c, 3 = sa, 4 = sb, 5 = sc
            r1 = tau1 * numpy.random.normal(0, 1)
            a_offspring = mi[i][0] + mi[i][3] * numpy.random.normal(0, 1)
            sa_offspring = mi[i][3] * math.exp(r1) * math.exp(tau2 * numpy.random.normal(0, 1))
            b_offspring = mi[i][1] + mi[i][4] * numpy.random.normal(0, 1)
            sb_offspring = mi[i][4] * math.exp(r1) * math.exp(tau2 * numpy.random.normal(0, 1))
            c_offspring = mi[i][2] + mi[i][5] * numpy.random.normal(0, 1)
            sc_offspring = mi[i][5] * math.exp(r1) * math.exp(tau2 * numpy.random.normal(0, 1))

            lambda_pop = numpy.append(lambda_pop, [[a_offspring, b_offspring, c_offspring,
                                                    sa_offspring, sb_offspring, sc_offspring,
                                                    mse_calc(a_offspring, b_offspring, c_offspring)]], axis=0)

    # check E_o - E_p:

    mi_errors = numpy.take(mi, [6], axis=1)
    min_mi_error_index = numpy.where(mi_errors == numpy.amin(mi_errors))
    # print(min_mi_error_index[0][0]) # index of the best parent
    lambda_pop_errors = numpy.take(lambda_pop, [6], axis=1)
    min_lambda_pop_error_index = numpy.where(lambda_pop_errors == numpy.amin(lambda_pop_errors))
    # print(min_lambda_pop_error_index[0][0])  #index of the best orrspring

    #poe = parent_offspring_error()
    #if poe != -1:
    #    print('found!!!')
    #    best_abcE = [lambda_pop[poe][0], lambda_pop[poe][1], lambda_pop[poe][2], lambda_pop[poe][6]]
    #    break  # wylezc z petli

    if abs(lambda_pop[min_lambda_pop_error_index[0][0]][6] - mi[min_mi_error_index[0][0]][6]) <= d:
        best_child = lambda_pop[min_lambda_pop_error_index[0][0]]
        print('found!!!')
        # best_abcE = [best_child[0], best_child[1], best_child[2], best_child[6]]
        # break  # wylezc z petli

    mi_and_lambda = numpy.append(mi, lambda_pop, axis=0)
    mi_and_lambda.view('i8, i8, i8, i8, i8, i8, i8').sort(order=['f6'], axis=0)
    mi = mi_and_lambda[0: mi_size]

    if best_abcE is None or mi[0][6] < best_abcE[3]:
        best_abcE = [mi[0][0], mi[0][1], mi[0][2], mi[0][6]]
        t_best = t
    t = t + 1

print("Best [a,b,c] = [" + str(best_abcE[0]) +', ' + str(best_abcE[1]) +', ' + str(best_abcE[2]) + ']')
print("MSE = " + str(best_abcE[3]))
print("iteration: " + str(t))

# plot of data 8 or 9:
fig, ax = plt.subplots()
ax.plot(numpy.take(data, [0], axis=1), numpy.take(data, [1], axis=1))  # i o
ax.set(xlabel='input', ylabel='output',
       title='original data from file')
ax.grid()
fig.savefig("data8.png")

# plot of data o_hat vs i
fig, ax = plt.subplots()
ax.plot(numpy.take(data, [0], axis=1), [best_abcE[0] * (numpy.square(row[0]) - best_abcE[1]*math.cos(best_abcE[1] * numpy.pi * row[0])) for row in numpy.take(data, [0], axis=1)])
ax.set(xlabel='input', ylabel='output',
       title='o_hat vs input')
ax.grid()
fig.savefig("out_8.png")



# todo:
#  ?????
#  what next?
#  it is very slow
