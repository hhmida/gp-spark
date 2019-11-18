#!/usr/bin/env python
import random
import operator
import itertools
import numpy
import time
import math
import sys
from pyspark import SparkContext, SparkConf
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

conf = SparkConf().setAppName("GP over Spark cluster")
sc = SparkContext(conf=conf)



## Preparing RDDs

TrainingRDD = sc.textFile('hdfs://nodemaster:9000/user/hadoop/HIGGS_Training_Scaled_10500.csv').map(lambda line : [float(x) for x in line.split(',')]).cache()
TestRDD = sc.textFile('hdfs://nodemaster:9000/user/hadoop/HIGGS_Test_Scaled_500.csv').map(lambda line : [float(x) for x in line.split(',')]).cache()


## Definig GP parameters

toolbox = base.Toolbox()
def sigmoid(x):
    return 1/(1+math.exp(-x))

### evaluating an individual
def evalHiggsBase(individual):
    
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    
    # Evaluate the sum of correctly identified as signal
    result = sum(TrainingRDD.map(lambda line: bool(sigmoid(func(*line[1:]))>0.5) is bool(line[0])).collect())
    
    return result,

# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2

    # Define a protected division function
def protectedDiv(left, right):
    try: 
        return left / right
    except ZeroDivisionError:
        return 1

# 28 iput float attributes
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 28), bool, "IN")

### Functions set

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# floating point operators
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)

# logic operators
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# float constant and boolean constants 
pset.addEphemeralConstant("rand1", lambda: random.random(), float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

#Set the best fitness as the max fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

#other parameters: selection, mutation, crossover, ....
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalHiggsBase)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


## Confusion matrix for evaluation

def confusionMatrix(func,dataset):
    confusion_matrix = [[0.0,0.0],[0.0,0.0]]
    predictions=dataset.map(lambda line:[bool(sigmoid(func(*line[1:]))>0.5),bool(line[0])]).collect()
    for line in predictions:
        predicted = line[0]
        real = line[1]
        if(predicted == real and real):
            confusion_matrix[0][0]+=1
        elif(predicted == real and not(real)):
            confusion_matrix[1][1]+=1
        elif(predicted != real and real):
            confusion_matrix[1][0]+=1
        elif(predicted != real and not(real)):
            confusion_matrix[0][1]+=1
    return confusion_matrix   

##  Distribution over Spark cluster

### Evaluating the whole population on a unique exeplar
def evalPop(individuals, line):
    return [bool(sigmoid(i(*[float(v) for v in line[1:]]))>0.5) is bool(float(line[0])) for i in individuals]

### Redefinig the GP loop from DEAP
def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    ii = [toolbox.compile(f) for f in invalid_ind]
    
    # Map and Reduce calls to evaluate the population
    results = TrainingRDD.map(lambda line: evalPop(ii, line))
    fitnesses = results.reduce(lambda v1,v2:list(map(operator.add,v1,v2)))
    fitnesses = [tuple([vf]) for vf in fitnesses]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluation of current generation
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        ii = [toolbox.compile(f) for f in invalid_ind]
        results = TrainingRDD.map(lambda line: evalPop(ii, line))
        fitnesses = results.reduce(lambda v1,v2:list(map(operator.add,v1,v2)))
        fitnesses = [tuple([vf]) for vf in fitnesses]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

    return population, logbook

## Redefinition
algorithms.eaSimple=eaSimple

## main program
def main():
    log=[]
    logRuns=[]
    nbRuns = int(sys.argv[1])
    popSize=int(sys.argv[2])
    nbGen = int(sys.argv[3])
    crossover_prob = 0.9
    mutation_prob = 0.04
    try:        
        for i in range(1,nbRuns+1):
            log.append('Run {} Using full training set'.format(i))
            random.seed()
            pop = toolbox.population(n=popSize)
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", numpy.mean)
            stats.register("std", numpy.std)
            stats.register("min", numpy.min)
            stats.register("max", numpy.max)
            log.append('Population size: {}'.format(popSize))
            log.append('Generations: {}'.format(nbGen))
            
            
            
            startTime = time.time()
            p,lbook = eaSimple(pop, toolbox, crossover_prob, mutation_prob, nbGen, stats, halloffame=hof, verbose=True)
            endTime = time.time()
            log.append("Training time: {} seconds".format(endTime-startTime))
            
            # Best individual
            expr = hof[0]
            
            # Best individual against full training set
            func = toolbox.compile(expr)
            confusion_matrix = confusionMatrix(func,TrainingRDD)
            resultTraining = confusion_matrix[0][0]+confusion_matrix[1][1]
            log.append(str(expr))
            log.append("fitness of best individual against total training set :{}/{}={}".format(resultTraining, numpy.sum(confusion_matrix), float(resultTraining) / numpy.sum(confusion_matrix)))
            cm = '\n'.join('\t'.join('%0.3f' %x for x in y) for y in confusion_matrix)
            log.append(cm)
            TP = confusion_matrix[0][0]
            TN = confusion_matrix[1][1]
            FP = confusion_matrix[0][1]
            FN = confusion_matrix[1][0]
            log.append("accuracy ={}, TPR={}, FPR={}".format((TP+TN)/(TP+TN+FP+FN),TP/(TP+FN),FP/(FP+TN)))
            
            # Best individual against test dataset
            confusion_matrix = confusionMatrix(func,TestRDD)
            resultTest = confusion_matrix[0][0]+confusion_matrix[1][1]
            
            log.append("\nfitness of best individual against total test set :{}/{}={}".format(resultTest, numpy.sum(confusion_matrix), float(resultTest) / numpy.sum(confusion_matrix)))
            cm = '\n'.join('\t'.join('%0.3f' %x for x in y) for y in confusion_matrix)
            log.append("\n"+cm)
            TP = confusion_matrix[0][0]
            TN = confusion_matrix[1][1]
            FP = confusion_matrix[0][1]
            FN = confusion_matrix[1][0]
            log.append("\naccuracy ={}, TPR={}, FPR={}".format((TP+TN)/(TP+TN+FP+FN),TP/(TP+FN),FP/(FP+TN)))
            # save current run log in sepperate file
            sc.parallelize(log).saveAsTextFile('FSS - run '+str(i)+'.log')
            logRuns.extend(log)
            logRuns.append("====================================================================================")
            del log[:]
    except Exception as e:
        logRuns.append('Exception: {}'.format(e))
    # save all runs log in higgs.log
    sc.parallelize(logRuns).saveAsTextFile('higgs.log')
    return pop, stats, hof

if __name__ == "__main__":
    error=[]
    try:
        main()
    except Exception as e:
        error.append("Exception: {}".format(e))
        sc.parallelize(error).saveAsTextFile('error.log')