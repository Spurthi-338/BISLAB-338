import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Objective function: given hyperparameter C, return mean cross-validation accuracy
def objective_function(C):
    # Create SVM with given C parameter
    model = SVC(C=C, kernel='rbf', gamma='scale')
    # Use 5-fold cross-validation to evaluate accuracy
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()

# PSO function (modified for single dimension hyperparameter search)
def pso(obj_func, bounds, num_particles=10, max_iter=30, w=0.7, c1=1.5, c2=1.5):
    dim = len(bounds)
    positions = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (num_particles, dim))
    velocities = np.zeros_like(positions)
    
    pbest_positions = np.copy(positions)
    pbest_values = np.array([obj_func(pos[0]) for pos in positions])
    
    gbest_index = np.argmax(pbest_values)
    gbest_position = pbest_positions[gbest_index]
    gbest_value = pbest_values[gbest_index]
    
    for iteration in range(max_iter):
        for i in range(num_particles):
            fitness = obj_func(positions[i][0])
            
            if fitness > pbest_values[i]:
                pbest_values[i] = fitness
                pbest_positions[i] = positions[i]
                
            if fitness > gbest_value:
                gbest_value = fitness
                gbest_position = positions[i]
        
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], [b[0] for b in bounds], [b[1] for b in bounds])
        
        print(f"Iteration {iteration+1}/{max_iter}, Best CV Accuracy: {gbest_value:.4f}, Best C: {gbest_position[0]:.4f}")
    
    return gbest_position, gbest_value

# Run PSO to find the best C hyperparameter for SVM
best_pos, best_val = pso(objective_function, bounds=[(0.1, 10)], num_particles=15, max_iter=30)

print("\nFinal Best C:", best_pos[0])
print("Final Best CV Accuracy:", best_val)
