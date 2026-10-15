"""
EV Platoon Optimization - Grey Wolf Optimizer (GWO) Comprehensive Comparison
MCTR 1021 - Performance Analysis Across Multiple Case Studies
Cleaned and fully functional version
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import csv

np.random.seed(42)

# ================= OBJECTIVE FUNCTIONS =================

def compute_beta(d, d_safe, d_range):
    beta = np.zeros_like(d, dtype=float)
    d = np.array(d, dtype=float)
    d_safe = np.array(d_safe, dtype=float)
    d_range = np.array(d_range, dtype=float)
    for i in range(len(d)):
        if d[i] <= d_safe[i]:
            beta[i] = 1.0
        elif d[i] <= d_range[i]:
            beta[i] = 1.0 - (d[i] - d_safe[i]) / (d_range[i] - d_safe[i])
        else:
            beta[i] = 0.0
    return beta

def total_energy_f1(masses, a, v, d, d_safe, d_range):
    beta = compute_beta(d, d_safe, d_range)
    masses = np.array(masses)
    a = np.array([a])
    v = np.array([v])
    energy = np.sum(masses * np.abs(a) * v * (1 - 0.5 * beta))
    return energy / len(masses)

def total_travel_time_f2(v, L, eps=1e-3):
    v_safe = max(v, eps)
    return L / v_safe

def total_number_of_vehicles_f3(N):
    return N

def inter_vehicle_distance_f4(d, K=1):
    d = np.array([d])
    N = len(d) + 1
    return np.sum(d) / (K * (N - 1))

def total_acceleration_effort_f5(a, K=1):
    a = np.array([a])
    N = len(a)
    return np.sum(np.abs(a)) / (K * N)

def checkConstraints(gene, Vmin, Vmax, Amin, Amax, SOCmin, SOCmax, dmin, dmax):
    velocity, acceleration, batteryUsage, distance, numOfCars = gene
    if not (Vmin <= velocity <= Vmax): return False
    if not (Amin <= acceleration <= Amax): return False
    if not (SOCmin <= batteryUsage <= SOCmax): return False
    if not (dmin <= distance <= dmax): return False
    if numOfCars < 2: return False
    return True

def calculate_fitness(solution, masses, L, d_safe, d_range, w1, w2, w3, w4, w5, K=1):
    velocity, acceleration, _, distance, numOfCars = solution
    f1 = total_energy_f1(masses, acceleration, velocity, [distance], d_safe, d_range)
    f2 = total_travel_time_f2(velocity, L)
    f3 = total_number_of_vehicles_f3(numOfCars)
    f4 = inter_vehicle_distance_f4([distance], K)
    f5 = total_acceleration_effort_f5([acceleration], K)
    # Normalize
    f1_norm = f1 / 50000
    f2_norm = f2 / 1000
    f3_norm = f3 / 15
    f4_norm = f4 / 50
    f5_norm = f5 / 4
    fitness = (w1*f1_norm + w2*f2_norm + w3*(1 - f3_norm) + w4*f4_norm + w5*f5_norm)
    return fitness, (f1,f2,f3,f4,f5)

# ================= GWO ALGORITHM =================

def initialize_population_gwo(pop_size, dim, bounds, Vmin, Vmax, Amin, Amax, SOCmin, SOCmax, dmin, dmax):
    population = []
    attempts = 0
    max_attempts = pop_size*100
    while len(population) < pop_size and attempts < max_attempts:
        attempts +=1
        wolf = []
        for low, high in bounds:
            wolf.append(np.random.uniform(low, high) if isinstance(low,float) else np.random.randint(low, high+1))
        if checkConstraints(wolf, Vmin, Vmax, Amin, Amax, SOCmin, SOCmax, dmin, dmax):
            population.append(wolf)
    return np.array(population)

def gwo(pop_size, dim, bounds, alpha_coeff, max_iter, masses, L, d_safe, d_range, w1,w2,w3,w4,w5,
        Vmin, Vmax, Amin, Amax, SOCmin, SOCmax, dmin, dmax, K=1):
    population = initialize_population_gwo(pop_size, dim, bounds, Vmin, Vmax, Amin, Amax, SOCmin, SOCmax, dmin, dmax)
    fitness = np.array([calculate_fitness(w, masses,L,d_safe,d_range,w1,w2,w3,w4,w5,K)[0] for w in population])
    
    idx_sorted = np.argsort(fitness)
    Alpha, Beta, Delta = population[idx_sorted[:3]]
    Alpha_fitness = fitness[idx_sorted[0]]
    
    fitness_history = []
    
    for t in range(max_iter):
        a = 2 - t*(2/max_iter)
        for i in range(pop_size):
            for j in range(dim):
                r1,r2 = np.random.rand(), np.random.rand()
                A1 = 2*a*r1 - a
                C1 = 2*r2
                D_alpha = abs(C1*Alpha[j] - population[i,j])
                X1 = Alpha[j] - A1*D_alpha
                
                r1,r2 = np.random.rand(), np.random.rand()
                A2 = 2*a*r1 - a
                C2 = 2*r2
                D_beta = abs(C2*Beta[j] - population[i,j])
                X2 = Beta[j] - A2*D_beta
                
                r1,r2 = np.random.rand(), np.random.rand()
                A3 = 2*a*r1 - a
                C3 = 2*r2
                D_delta = abs(C3*Delta[j] - population[i,j])
                X3 = Delta[j] - A3*D_delta
                
                population[i,j] = np.clip((X1+X2+X3)/3, bounds[j][0], bounds[j][1])
            
            if not checkConstraints(population[i], Vmin, Vmax, Amin, Amax, SOCmin, SOCmax, dmin, dmax):
                population[i] = np.random.uniform([b[0] for b in bounds],[b[1] for b in bounds])
        
        fitness = np.array([calculate_fitness(w, masses,L,d_safe,d_range,w1,w2,w3,w4,w5,K)[0] for w in population])
        idx_sorted = np.argsort(fitness)
        Alpha, Beta, Delta = population[idx_sorted[:3]]
        Alpha_fitness = fitness[idx_sorted[0]]
        fitness_history.append(Alpha_fitness)
    
    return Alpha, Alpha_fitness, fitness_history

# ================= CASE STUDIES =================

def get_case_studies_gwo():
    return {
        1: {'name':'Small Urban Platoon',
            'bounds': [(10.0, 20.0), (-2.0,2.0), (30.0,100.0), (5.0,20.0), (2,5)],
            'Vmin':10,'Vmax':20,'Amin':-2,'Amax':2,'SOCmin':30,'SOCmax':100,'dmin':5,'dmax':20,
            'masses': np.array([1500,1550,1600]), 'L':5000.0, 'weights':(0.4,0.25,0.15,0.1,0.1),
            'pop_size':30,'max_iter':200,'alpha_coeff':2},
        2: {'name':'Medium Highway Platoon',
            'bounds': [(15.0,30.0),(-3.0,3.0),(20.0,100.0),(10.0,40.0),(3,8)],
            'Vmin':15,'Vmax':30,'Amin':-3,'Amax':3,'SOCmin':20,'SOCmax':100,'dmin':10,'dmax':40,
            'masses':np.array([1500,1550,1600,1650,1700]),'L':10000.0,'weights':(0.3,0.3,0.2,0.1,0.1),
            'pop_size':50,'max_iter':200,'alpha_coeff':2},
        3: {'name':'Large Highway Platoon',
            'bounds':[(20.0,35.0),(-3.0,3.0),(20.0,100.0),(15.0,50.0),(5,15)],
            'Vmin':20,'Vmax':35,'Amin':-3,'Amax':3,'SOCmin':20,'SOCmax':100,'dmin':15,'dmax':50,
            'masses':np.array([1500,1550,1600,1650,1700,1750,1800]),'L':20000.0,'weights':(0.25,0.25,0.25,0.15,0.1),
            'pop_size':80,'max_iter':200,'alpha_coeff':2}
    }

# ================= MULTIPLE TRIALS =================

def run_gwo_trials(case, num_trials=10):
    fitness_histories = []
    best_fitness_list = []
    time_list = []
    
    for t in range(num_trials):
        print(f"  Trial {t+1}/{num_trials}...", end=' ')
        start = time.time()
        Alpha, Alpha_fit, hist = gwo(
            case['pop_size'], 5, case['bounds'], case['alpha_coeff'], case['max_iter'],
            case['masses'], case['L'], np.array([8.0]), np.array([15.0]), *case['weights'],
            case['Vmin'], case['Vmax'], case['Amin'], case['Amax'],
            case['SOCmin'], case['SOCmax'], case['dmin'], case['dmax']
        )
        elapsed = time.time() - start
        print(f"Fitness: {Alpha_fit:.4f}, Time: {elapsed:.2f}s")
        
        fitness_histories.append(hist)
        best_fitness_list.append(Alpha_fit)
        time_list.append(elapsed)
        
        save_trial_log(hist, case['name'], t)
    
    mean_fitness = np.mean(best_fitness_list)
    std_fitness = np.std(best_fitness_list)
    
    return {
        'fitness_histories': fitness_histories,
        'best_fitness_list': best_fitness_list,
        'time_list': time_list,
        'mean_fitness': mean_fitness,
        'std_fitness': std_fitness,
        'best_of_all': min(best_fitness_list),
        'worst_of_all': max(best_fitness_list)
    }

# ================= PLOTTING =================

def plot_convergence(fitness_histories, case_name, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8,6))
    for i, hist in enumerate(fitness_histories):
        plt.plot(hist, alpha=0.6, label=f'Trial {i+1}')
    plt.title(f'GWO Convergence Curves - {case_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f'gwo_{case_name}_convergence.png'))
    plt.close()

def plot_boxplot(best_fitness_list, case_name, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.boxplot(best_fitness_list, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title(f'GWO Best Fitness Distribution - {case_name}')
    plt.ylabel('Best Fitness')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f'gwo_{case_name}_boxplot.png'))
    plt.close()

def plot_summary(case_results, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)
    case_names = [case_results[k]['name'] for k in case_results]
    mean_fitness = [case_results[k]['mean_fitness'] for k in case_results]
    std_fitness = [case_results[k]['std_fitness'] for k in case_results]
    
    x = np.arange(len(case_names))
    plt.figure(figsize=(8,6))
    plt.bar(x, mean_fitness, yerr=std_fitness, capsize=5, color='lightgreen')
    plt.xticks(x, case_names, rotation=15)
    plt.ylabel('Mean Best Fitness ± Std')
    plt.title('GWO Performance Summary Across Cases')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'gwo_summary.png'))
    plt.close()

# ================= LOGGING =================

def save_trial_log(fitness_history, case_name, trial_idx, save_dir='logs'):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'gwo_{case_name}_trial{trial_idx}.csv')
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration','BestFitness'])
        for i, fit in enumerate(fitness_history):
            writer.writerow([i+1, fit])

def save_summary_csv(results_all, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir,'gwo_summary.csv')
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Case','MeanBestFitness','StdBestFitness','BestOverallFitness'])
        for key, res in results_all.items():
            writer.writerow([res['name'], res['mean_fitness'], res['std_fitness'], res['best_of_all']])

# ================= FULL REPORT =================

def gwo_full_report(case_studies, num_trials=10):
    results_all = {}
    for key, case in case_studies.items():
        print(f"\n=== CASE {key}: {case['name']} ===")
        results = run_gwo_trials(case, num_trials)
        results['name'] = case['name']
        results_all[key] = results
        
        plot_convergence(results['fitness_histories'], case['name'])
        plot_boxplot(results['best_fitness_list'], case['name'])
    
    save_summary_csv(results_all)
    plot_summary(results_all)
    
    return results_all

# ================= MAIN =================

def main():
    case_studies = get_case_studies_gwo()
    gwo_results = gwo_full_report(case_studies, num_trials=10)
    # If SA and GA results are available, you can call:
    # save_comparison_csv(sa_results_all, ga_results_all, gwo_results)

if __name__ == "__main__":
    main()
