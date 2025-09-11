import random
import math

def fitness(x):
    return x * math.sin(10 * math.pi * x) + 2

def random_gene():
    return random.uniform(-1, 2)

def express(chromosome):
    return sum(chromosome) / len(chromosome)

def create_chromosome(length):
    return [random_gene() for _ in range(length)]

def initialize_population(size, gene_length):
    return [create_chromosome(gene_length) for _ in range(size)]

def select(population, fitnesses):
    total = sum(fitnesses)
    r = random.uniform(0, total)
    cumulative = 0
    for chrom, fit in zip(population, fitnesses):
        cumulative += fit
        if cumulative >= r:
            return chrom
    return random.choice(population)

def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1[:], parent2[:]

def mutate(chromosome, mutation_rate):
    return [
        random_gene() if random.random() < mutation_rate else gene
        for gene in chromosome
    ]

def gene_expression_algorithm():
    POP_SIZE = 6
    GENE_LENGTH = 10
    MUTATION_RATE = 0.05
    CROSSOVER_RATE = 0.8
    GENERATIONS = 20

    population = initialize_population(POP_SIZE, GENE_LENGTH)
    best_chromosome = None
    best_fitness = float('-inf')

    for gen in range(GENERATIONS):
        fitnesses = [fitness(express(chrom)) for chrom in population]

        for chrom, fit in zip(population, fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_chromosome = chrom[:]

        print(f"Generation {gen+1}: Best Fitness = {best_fitness:.4f}, x = {express(best_chromosome):.4f}")

        new_population = []
        while len(new_population) < POP_SIZE:
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            child1, child2 = crossover(parent1, parent2, CROSSOVER_RATE)
            child1 = mutate(child1, MUTATION_RATE)
            child2 = mutate(child2, MUTATION_RATE)
            new_population.extend([child1, child2])
        population = new_population[:POP_SIZE]

    x_best = express(best_chromosome)
    print("\nBest solution found:")
    print(f"Genes: {best_chromosome}")
    print(f"x = {x_best:.4f}")
    print(f"f(x) = {fitness(x_best):.4f}")

if __name__ == "__main__":
    gene_expression_algorithm()
