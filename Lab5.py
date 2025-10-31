import numpy as np

def sphere_function(x):
    return np.sum(x**2)

class GreyWolfOptimizer:
    def __init__(self, obj_func, dim, lb, ub, population=20, max_iter=100):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.population = population
        self.max_iter = max_iter
        
        # Initialize wolves positions randomly within boundaries
        self.positions = np.random.uniform(lb, ub, (population, dim))
        self.fitness = np.full(population, np.inf)
        
        # Initialize alpha, beta, delta wolves
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = np.inf
        
        self.beta_pos = np.zeros(dim)
        self.beta_score = np.inf
        
        self.delta_pos = np.zeros(dim)
        self.delta_score = np.inf
    
    def optimize(self):
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # linearly decreasing a from 2 to 0
            
            for i in range(self.population):
                # Evaluate fitness
                fitness = self.obj_func(self.positions[i])
                self.fitness[i] = fitness
                
                # Update alpha, beta, delta
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()
            
            for i in range(self.population):
                for d in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    D_alpha = abs(C1 * self.alpha_pos[d] - self.positions[i, d])
                    X1 = self.alpha_pos[d] - A1 * D_alpha
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    D_beta = abs(C2 * self.beta_pos[d] - self.positions[i, d])
                    X2 = self.beta_pos[d] - A2 * D_beta
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    D_delta = abs(C3 * self.delta_pos[d] - self.positions[i, d])
                    X3 = self.delta_pos[d] - A3 * D_delta
                    
                    # Update position for wolf i at dimension d
                    self.positions[i, d] = (X1 + X2 + X3) / 3
                
                # Boundary check
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
        
        return self.alpha_pos, self.alpha_score


# Usage example
dim = 5
lb = -10
ub = 10
gwo = GreyWolfOptimizer(sphere_function, dim, lb, ub, population=30, max_iter=100)
best_position, best_score = gwo.optimize()

print("Best solution found:", best_position)
print("Best objective value:", best_score)
