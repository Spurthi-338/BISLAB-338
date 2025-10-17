import numpy as np
import math

def objective_function(x):
    # Example: Sphere function (minimization)
    return np.sum(x ** 2)

def levy_flight(Lambda, size):
    sigma1 = (math.gamma(1 + Lambda) * math.sin(np.pi * Lambda / 2) /
              (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size)
    v = np.random.normal(0, sigma2, size)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step

def cuckoo_search(obj_func, n=15, d=2, max_iter=100, pa=0.25, alpha=0.01, bounds=(-5, 5)):
    lower, upper = bounds
    nests = np.random.uniform(lower, upper, (n, d))
    fitness = np.array([obj_func(x) for x in nests])
    best_idx = np.argmin(fitness)
    best_solution = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    for t in range(max_iter):
        for i in range(n):
            step = levy_flight(1.5, d)
            new_solution = nests[i] + alpha * step * (nests[i] - nests[np.random.randint(n)])
            new_solution = np.clip(new_solution, lower, upper)
            new_fitness = obj_func(new_solution)
            if new_fitness < fitness[i]:
                nests[i] = new_solution
                fitness[i] = new_fitness

        K = np.random.rand(n) > pa
        stepsize = np.random.rand(n, d) * (nests[np.random.permutation(n)] - nests[np.random.permutation(n)])
        new_nests = nests + stepsize * K[:, None]
        new_nests = np.clip(new_nests, lower, upper)

        new_fitness = np.array([obj_func(x) for x in new_nests])
        improved = new_fitness < fitness
        nests[improved] = new_nests[improved]
        fitness[improved] = new_fitness[improved]

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution = nests[best_idx].copy()
            best_fitness = fitness[best_idx]

    return best_solution, best_fitness

# Example run
best_sol, best_fit = cuckoo_search(objective_function, n=20, d=3, max_iter=200)
print("Best Solution:", best_sol)
print("Best Fitness:", best_fit)
