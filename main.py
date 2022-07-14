import numpy as np
from statistics import mean
import random
import matplotlib.pyplot as plt


#inicializar população
populations =([[random.randint(0,1) for x in range(5)] for i in range(4)]) #random.randint(a, b): Return a random integer N such that a <= N <= b
#populations = [[0,1,1,0,1],[1,1,0,0,0],[0,1,0,0,0],[1,0,0,1,1]] #descomentar em caso de querer encontrar o mesmo resultado do livro
print("population:",populations)
L=len(populations)
LL=len(populations[0])-1

# função que serve para calcular o primeiro x value e fitness encontrados, além de prob, expected count, actual count e somatório, média e o máximo de fitness, prob, expected count e actual count
def fitness_score():
    global populations
    global act
    fit_value = []
    x = []
    probx2 = []
    expected = []
    actual = []
    total1 = 0
    total2 = 0

    print("                                       PRIMEIRA ETAPA                                       ")

    #a partir do chromosome_value (binário), se encontra o x value (inteiro) correspondente
    for i in range(L):
        chromosome_value = 0

        for j in range(L, -1, -1):
            chromosome_value += populations[i][j]*(2**(LL-j))  #pow(2,4-j) ou 2ˆ(4-j)

        # Append does not provide any return value. So if you add an item to a list with append, it will not return the list; it will merely alter the list and add the item in the end
        fit_value.append(chromosome_value ** 2) #f(x)=xˆ2 -> fitness
        x.append(chromosome_value)


    for i in range(L):
        prob=fit_value[i]/sum(fit_value)
        probx2.append(prob)

        expected_count=fit_value[i]/mean(fit_value)
        expected.append(expected_count)

    print("x value:", x)
    print("fitness:", fit_value)
    print("prob:", probx2)
    print("expected count:", expected) #The expected count gives an idea of which population can be selected for further processing in the mating pool.

    #The actual count of string no 0 is 1, hence it occurs once in the mating pool. The actual count of string no 1 is 2, hence it occurs
    #twice in the mating pool. Since the actual count of string no 2 is 0,
    #it does not occur in the mating pool. Similarly, the actual count of
    #string no 3 being 1, it occurs once in the mating pool.
    for i in range(L):
        actual_count=round(expected[i]) #calculando o actual count a partir do arredondamento do expected count
        actual.append(actual_count)

    act = actual
    print("actual count:", actual)

    #calculando os máximos de cada parâmetro da tabela
    max_fitness = max(fit_value)
    max_prob = max(probx2)
    max_expected = max(expected)
    max_actual = max(actual)

    print("sum_fitness:", sum(fit_value))
    print("mean_fitness:", mean(fit_value))
    print("max_fitness:", max_fitness)

    for i in range(L): #para variavel prob float é necessario fazer assim
        total1 += probx2[i]

    print("sum_prob:", total1)
    print("mean_prob:", total1/len(probx2))
    print("max_prob:", max_prob)

    for i in range(L): #para variavel expected count float é necessario fazer assim
        total2 += expected[i]

    print("sum_expected count:", total2)
    print("mean_expected count:", total2 / len(expected))
    print("max_expected count:", max_expected)

    print("sum_actual count:", sum(actual))
    print("mean_actual count:", mean(actual))
    print("max_actual count:", max_actual)
    print("--------------------------------------------------------------------------------------------------------------------------------")

fitness_score()


# função que serve para calcular os x value e fitness depois de crossover e mutation
def fitness_score1():
    fit_value = []
    x = []
    global maxfit

    # chromosome_value=x -> x value
    for i in range(len(m3_pool)):
        chromosome_value = 0

        for j in range(len(m3_pool), -1, -1):
            chromosome_value += m3_pool[i][j] * (2 ** ((len(m3_pool[0])-1) - j))  # pow(2,4-j) ou 2ˆ(4-j)

        # Append does not provide any return value. So if you add an item to a list with append, it will not return the list; it will merely alter the list and add the item in the end
        fit_value.append(chromosome_value ** 2)  # f(x)=xˆ2 -> fitness
        x.append(chromosome_value)

    maxfit = max(fit_value)

    print("x value:", x)
    print("fitness:", fit_value)

    print("sum_fitness:", sum(fit_value))
    print("mean_fitness:", mean(fit_value))
    print("max_fitness:", max(fit_value))

def crossover(m3_pool,cross3):

    m5_pool = []
    for i in range (len(m3_pool)): #conversão de list para numpy
        m3_pool[i] = np.array(m3_pool[i])

    #o cruzamento por 1-point crossover é realizado trocando os bits do indivíduo/genótipo i com o i+1
    i=0
    j=0
    while(i<len(m3_pool)-1): #i<4-1 -> i<3
                j = i//2

                m4_pool = np.append(m3_pool[i][:cross3[j]], m3_pool[i+1][cross3[j]:])

                for k in range(len(m4_pool)):
                    m4_pool[k] = m4_pool[k].tolist()

                m5_pool.append(m4_pool)

                for k in range(len(m4_pool)):
                    m4_pool[k] = np.array(m4_pool[k])

                m4_pool = np.append(m3_pool[i+1][:cross3[j]], m3_pool[i][cross3[j]:])

                for k in range(len(m4_pool)):
                    m4_pool[k] = m4_pool[k].tolist()

                m5_pool.append(m4_pool)

                i=i+2

    for i in range(len(m5_pool)): #conversão de numpy para list
        m5_pool[i] = m5_pool[i].tolist()

    return m5_pool

def matpool_crosspoint():
    global cross3
    global m3_pool

    m2_pool = []
    for i in range(len(populations)):
            if act[i]!=0 and act[i]!=1:
                m_pool = populations[i] * act[i]
                m_pool = np.array(m_pool)
                m_pool = np.array_split(m_pool, act[i])
                for j in range(act[i]):
                    m_pool[j] = m_pool[j].tolist()
                    m2_pool.append(m_pool[j])
            else:
                m_pool = populations[i] * act[i]
                m2_pool.append(m_pool)
    m2_pool = [e for e in m2_pool if e] #excluir arrays vazios
    print("                                       SEGUNDA ETAPA                                       ")
    print("mating pool:", m2_pool)

    m3_pool = m2_pool

    #escolher os pontos de cruzamento em cada linha do mating pool
    cross2 = []
    for i in range(len(m2_pool)//2): #'float' object cannot be interpreted as an integer" occurs when we pass a float to a function that expects an integer argument. To solve the error, use the floor division operator, e.g. for i in range(my_num // 5)
        cross = random.randint(0,len(m2_pool[0])-1)
        cross2.append(cross)
    print("cross:", cross2)

    cross3 = cross2

    #cross3 = [4,2] #descomentar em caso de querer encontrar o mesmo resultado do livro
    print(cross3)
    m3_pool = crossover(m3_pool,cross3)
    print("offspring after xover:", m3_pool)

    fitness_score1()

matpool_crosspoint()

def mutation():
    print(".................................................................................................................................")
    print("                                       TERCEIRA ETAPA                                       ")
    #mutation_rate=0.1 #10%
    mutation_rate=0.1 #1%
    #mutation_rate=0.05 #5%
    #mutation_rate=0.005 #0.5%
    #mutation_rate=0.001 #0.1% #descomentar em caso de querer utilizar a mesma taxa de mutação recomendada pelo livro

    rag = len(m3_pool)*len(m3_pool[0])

    bit_rate = ([[random.uniform(0,1) for x in range(len(m3_pool[0]))] for i in range(len(m3_pool))])
    print("bit_rate:", bit_rate)

    for i in range(len(m3_pool)):
        for j in range(len(m3_pool[i])):
            if bit_rate[i][j]<mutation_rate:
                m3_pool[i][j]=1-m3_pool[i][j]

    #descomentar em caso de querer obter o mesmo resultado do livro
    #m3_pool[0][0]=1-m3_pool[0][0]
    #m3_pool[3][2]=1-m3_pool[3][2]

    fitness_score1()

    print("offspring after mutation", m3_pool)
    populations = m3_pool.copy()
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


mutation()

i=0
iii = []
fittt = []
while i<2: #loop para realizar diversas iterações até convergir para o melhor fitness value

    fitness_score()
    matpool_crosspoint()
    mutation()
    

    iii.append(i)
    fittt.append(maxfit)

    i=i+1

for k in range(len(iii)): #conversão de list para numpy
    iii[k] = np.array(iii[k])
    fittt[k] = np.array(fittt[k])


print("total_iterations_maxfitness:", max(fittt))
#index = np.amax(fittt)

#print("best population:", m3_pool[index])

#plotar os fitness values ao longo das gerações
plt.plot(iii, fittt, 'm')

plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.title('f(x)=xˆ2')
#plt.grid()
plt.show()




