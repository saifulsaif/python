import random

def Initialization(sizeOfPop, NUM_ITEMS):
    Population = []
    for x in range(0, sizeOfPop):
        individual = []
        for y in range(0, NUM_ITEMS):
            individual.append(random.randint(0, 1))
        Population.append(individual)
    return Population

def fitness(individual, ITEMS, KnapsackCapacity):
    total_value = 0
    total_weight = 0
    index = 0
    for i in individual:
        if index >= len(ITEMS):
            break
        if i == 1:
            total_value += ITEMS[index][0]
            total_weight += ITEMS[index][1]
        index += 1
    if total_weight > KnapsackCapacity:
        return 0
    else:
        return total_value

def Evolution(pop):
    parent_percent = 0.2
    mutation_chance = 0.08
    parent_lottery = 0.05
    
    # Selection
    parent_length = int(parent_percent * len(pop))
    parents = pop[:parent_length]
    
    # CROSS-OVER
    children = []
    desired_length = len(pop) - len(parents)
    while len(children) < desired_length:
        male = pop[random.randint(0, len(parents) - 1)]
        female = pop[random.randint(0, len(parents) - 1)]
        half = int(len(male) / 2)
        child = male[:half] + female[half:]
        children.append(child)
        
    # Mutation
    for child in children:
        if mutation_chance > random.random():
            r = random.randint(0, len(child) - 1)
            if child[r] == 1:
                child[r] = 0
            else:
                child[r] = 1
    parents.extend(children)
    return parents

def main():
    KCap = 50
    POP_SIZE = 30
    GEN_MAX = 50
    NUM_ITEMS = 15
    ITEMS = [(random.randint(0, 20), random.randint(0, 20)) for x in range(0, NUM_ITEMS)]
    
    generation = 1 # Generation counter
    population = Initialization(POP_SIZE, NUM_ITEMS)
    
    for g in range(0, GEN_MAX):
        population = sorted(population, key=lambda ind: fitness(ind, ITEMS, KCap), reverse=True)
        totalFitness = 0
        for i in population:
            totalFitness += fitness(i, ITEMS, KCap)
        print(f"Generation {generation} Total Fitness: {totalFitness}")
        
        population = Evolution(population)
        generation += 1
        
    population = sorted(population, key=lambda ind: fitness(ind, ITEMS, KCap), reverse=True)
    print("Best Configuration:", population[0])

if __name__ == "__main__":
    main()
