##############################################
#                  LIBRERIAS                 #
##############################################
from deap import base, creator, tools
import pickle
from .generadorFinal import JSP
import numpy as np
import time
import array
import operator 
import math
import os
from minizinc import Instance, Model, Solver
from datetime import timedelta
import json
import itertools
import random 

class GA2:
    def __init__(self,dir, population_size, generations, cross_rate, mutation_rate, seed=0) -> None:
        self.population_size = population_size
        self.generations     = generations
        self.cross_rate      = cross_rate
        self.mutation_rate   = mutation_rate

        with open(f'{dir}{dir.split("/")[-2]}','rb') as file:
            self.problema:JSP = pickle.load(file)

        # Fijamos la semilla
        np.random.seed(seed)    
        random.seed(seed)   

        # Used in generate_random_operation
        self.options = list(range(self.problema.numJobs))*self.problema.numMchs        
        np.random.shuffle(self.options)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initRepeat, creator.IndividualFloat, lambda: np.random.uniform(0,3), n=self.problema.numJobs * self.problema.numMchs*2)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.fitness)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selRoulette)

        # Población inicial
        self.population = self.toolbox.population(n=self.population_size)
        
        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit
    
    def fitness(self, individual):
        return self.problema.objective_function_solution(self.get_permutation(individual))[0],

    def get_permutation(self, particle):
        permutation = []

        job   = particle[::2]
        speed = particle[1::2]

        job_index_sorted = sorted(enumerate(zip(job, speed)), key=lambda x: x[1])
        for i, (job, speed )in job_index_sorted:
            permutation.append(i // self.problema.numMchs)
            permutation.append(min(max(0,round(speed)), self.problema.speed - 1))
        return permutation
    
    def cross(self, offspring):
        new_population = []
        for parent1, parent2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < self.cross_rate:
                point_cross    = np.random.randint(0, len(parent1))
                child1 = creator.IndividualFloat(parent1[:point_cross] + parent2[point_cross:])
                child2 = creator.IndividualFloat(parent2[:point_cross] + parent1[point_cross:])
                new_population.append(child1)
                new_population.append(child2)
        return new_population

    def mutate(self, offspring):
        for i,individual in enumerate(offspring):
            for gen, value in enumerate(individual):
                if np.random.random() < self.mutation_rate:
                    new_gene = value + 0.1 if np.random.random() < 0.5 else value - 0.1              
                    individual[gen] = max(min(new_gene,3),0) if gen % 2 == 0 else max(min(new_gene,self.problema.speed),0)
        return offspring

    def solve(self, path, timeout=None, replace=True):
        if not replace and os.path.exists(f'{path}-ga2.json'):
            return        
        
        iteraciones          = 0                    # Número de iteraciones realizadas
        tiempo_inicial       = time.time()          # Tiempo en el que empieza
        tiempo_transcurrido  = 0                    # Tiempo total transcurrido desde el inicio de la resolucion
        tiempo_por_iteracion = 0                    # Tiempo medio consumido por iteracion
        
        if not timeout:
            timeout = 60
        else:
            timeout = timeout[0][self.problema.numJobs-1] + timeout[0][self.problema.numMchs-1] + timeout[1][self.problema.rddd] + timeout[2][self.problema.speed-1]
            timeout = timeout / 1000

        # Condición de parada
        while(tiempo_transcurrido + tiempo_por_iteracion < timeout and iteraciones < self.generations):
            # Algoritmo genético
            # SELECCION
            selected       = self.toolbox.select(self.population, self.population_size)
            # CRUCE
            offspring      = self.cross(selected)
            # MUTACION
            offspring      = self.mutate(offspring)
            # Evaluate the offsprin fitnesses
            fitnesses = map(self.toolbox.evaluate, offspring)
            for ind, fit in zip(offspring,fitnesses):
                ind.fitness.values = fit
            new_population = self.population + offspring
            self.population = new_population

            # print(self.population)
            # quit()
            #print(min([sol.fitness.values for sol in self.population]))

            # Actualización de las variables de control del tiempo de ejecución
            iteraciones += 1
            tiempo_transcurrido = time.time() - tiempo_inicial
            tiempo_por_iteracion = tiempo_transcurrido / iteraciones  
        stats = {}
        best = tools.selBest(self.population, 1)[0]
        # print(best)  
        makespan, energy, tardiness   = self.problema.objective_function_solution(self.get_permutation(best))[1]
        stats["bestSolution"]         = self.get_permutation(best)
        stats["makespan"]             = int(makespan)
        stats["normalized_makespan"]  = self.problema.norm_makespan(makespan)
        stats["energy"]               = int(energy)
        stats["normalized_energy"]    = self.problema.norm_energy(energy)
        stats["tardiness"]            = int(tardiness)
        stats["normalized_tardiness"]  = self.problema.norm_tardiness(tardiness)
        stats["bestObjectiveFunction"]= best.fitness.values[0]
        stats["tiempo_por_iteracion"] = tiempo_por_iteracion
        
        with open(f'{path}-ga.json','w') as file:
            json.dump(stats, file, indent=4)

        # del creator.Individual
        # del creator.Fitness

        return best.fitness.values[0]
class GA:
    
    def generate_random_operation(self):
        if len(self.options) == 0:
            self.options = list(range(self.problema.numJobs))*self.problema.numMchs
            np.random.shuffle(self.options)
        return self.options.pop()

    def generate_random_speed(self):
        return np.random.choice(range(self.problema.speed))

    def __init__(self, dir, population_size, generations, cross_rate, mutation_rate, seed=0) -> None:
        self.population_size = population_size
        self.generations     = generations
        self.cross_rate      = cross_rate
        self.mutation_rate   = mutation_rate

        

        with open(f'{dir}{dir.split("/")[-2]}','rb') as file:
            self.problema:JSP = pickle.load(file)

        # print("Maximo Makespan: ", self.problema.max_min_makespan)
        # print("Maximo Energy  : ", self.problema.max_min_energy)
        # print("Diferencia     : ", self.problema.max_min_makespan - self.problema.max_min_energy)

        # Fijamos la semilla
        np.random.seed(seed) 
        random.seed(seed)      

        # Used in generate_random_operation
        self.options = list(range(self.problema.numJobs))*self.problema.numMchs        
        np.random.shuffle(self.options)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initCycle, creator.Individual, [self.generate_random_operation,
                         self.generate_random_speed], n=self.problema.numJobs * self.problema.numMchs)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.fitness)
        self.toolbox.register("mate", self.cross_operator)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=2)       

        # self.population = []

        # grasp = GRASP(DIR, self.population_size, 0.1, 100,seed)
        # for _ in range(self.population_size):
        #     solution, fitness = grasp.construction()
        #     cromo = creator.Individual(solution)
        #     cromo.fitness.values = fitness,
        #     self.population.append(cromo)
                
    
    def cross(self, parents):
        offspring = []
        for parent1, parent2 in zip(parents[::2], parents[1::2]):
            if np.random.random() < self.cross_rate:
                job     = np.random.randint(0,self.problema.numJobs - 1)
                child1  = self.toolbox.mate(parent1, parent2, job) 
                child2  = self.toolbox.mate(parent2, parent1, job)  
                offspring.append(child1)
                offspring.append(child2)
        return offspring
    
    def cross_operator(self, parent1:array.array, parent2:array.array, job:int):
        child  = array.array('i')
        puntero1 = 0
        puntero2 = 0
        cont = 0
        while puntero1 < len(parent1):
            if parent1[puntero1] == job:
                while cont > 0:
                    if parent2[puntero2] != job:
                        child.append(parent2[puntero2])
                        child.append(parent2[puntero2 + 1])
                        cont -= 1
                    puntero2 += 2
                child.append(parent1[puntero1])
                child.append(parent1[puntero1 + 1])
            else :
                cont += 1
            puntero1 += 2
        while puntero2 < len(parent2):
            if parent2[puntero2] != job:
                child.append(parent2[puntero2])
                child.append(parent2[puntero2 + 1])
            puntero2 += 2
        return creator.Individual(child)

    def generate_other_speed(self, speed):
        options = [i for i in range(self.problema.speed) if i != speed]
        new_speed = np.random.choice(options)
        return new_speed
    def mutate(self, offspring):
        new_offspring = list()
        for i,individual in enumerate(offspring):
            if np.random.random() < self.mutation_rate:
                new_individual = array.array('i',individual)
                # Two sets (task, energy) are randomly selected
                a,b = np.random.choice(range(0,len(individual), 2), 2, replace=True)
                a,b = min(a,b), max(a,b)
                mixed_pairs = list(range(a,b+2,2))
                np.random.shuffle(mixed_pairs)
                for i,index in enumerate(range(a,b+2,2)):
                    new_individual[index] = individual[mixed_pairs[i]]
                    new_individual[index + 1] = individual[mixed_pairs[i] + 1]
                if self.problema.speed > 1:
                    for i in range(1,len(new_individual),2):
                        new_individual[i] = self.generate_other_speed(new_individual[i])                
                new_offspring.append(creator.Individual(new_individual))
            else:
                new_offspring.append(individual)
        return new_offspring

    def fitness(self, individual):        
        return self.problema.objective_function_solution(individual)[0],

    def solve(self, path, timeout=None, replace=True, guardar=True):        
        if not replace and os.path.exists(f'{path}-ga.json'):
            return        
        
        # Población inicial
        self.population = self.toolbox.population(n=self.population_size)
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit
        
        iteraciones          = 0                    # Número de iteraciones realizadas
        tiempo_inicial       = time.time()          # Tiempo en el que empieza
        tiempo_transcurrido  = 0                    # Tiempo total transcurrido desde el inicio de la resolucion
        tiempo_por_iteracion = 0                    # Tiempo medio consumido por iteracion
        
        if not timeout:
            timeout = 60
        elif type(timeout) != float and type(timeout) != int:
            timeout = timeout[0][self.problema.numJobs-1] + timeout[0][self.problema.numMchs-1] + timeout[1][self.problema.rddd] + timeout[2][self.problema.speed-1]
            timeout = timeout / 1000

        # Condición de parada
        while(tiempo_transcurrido + tiempo_por_iteracion < timeout ):#and iteraciones < self.generations):
            # Algoritmo genético            
            # CRUCE
            offspring      = self.cross(self.population)
            # MUTACION
            offspring      = self.mutate(offspring)
            # Evaluate the offsprin fitnesses
            fitnesses = map(self.toolbox.evaluate, offspring)
            for ind, fit in zip(offspring,fitnesses):
                ind.fitness.values = fit
            new_population = self.population + offspring
            # SELECCION
            selected        = self.toolbox.select(new_population, self.population_size)
            self.population = selected
            
            #print(np.mean([sol.fitness.values[0] for sol in self.population]))
            # print(self.population)
            # quit()
            # print(min([sol.fitness.values for sol in self.population]))

            # Actualización de las variables de control del tiempo de ejecución
            iteraciones += 1
            tiempo_transcurrido = time.time() - tiempo_inicial
            tiempo_por_iteracion = tiempo_transcurrido / iteraciones  
        stats = {}
        best = tools.selBest(self.population, 1)[0]
        # print(best)  
        if guardar:
            makespan, energy, tardiness   = self.problema.objective_function_solution(best)[1]
            stats["bestSolution"]         = best.tolist()
            stats["makespan"]             = int(makespan)
            stats["normalized_makespan"]  = self.problema.norm_makespan(makespan)
            stats["energy"]               = int(energy)
            stats["normalized_energy"]    = self.problema.norm_energy(energy)
            stats["tardiness"]            = int(tardiness)
            stats["normalized_tardiness"]  = self.problema.norm_tardiness(tardiness)
            stats["bestObjectiveFunction"]= best.fitness.values[0]
            stats["tiempo_por_iteracion"] = tiempo_por_iteracion
            
            with open(f'{path}-ga.json','w') as file:
                json.dump(stats, file, indent=4)

            # del creator.Individual
            # del creator.Fitness

        return best.fitness.values[0]

class PSO:
    def __init__(self, dir, num_particles, iterations, c1, c2, w, seed = 0) -> None:
        
        self.num_particles = num_particles
        self.iterations    = iterations
        self.c1            = c1
        self.c2            = c2
        self.w             = w

        with open(f'{dir}{dir.split("/")[-2]}','rb') as file:
            self.problema:JSP = pickle.load(file)

        # Fijamos la semilla
        np.random.seed(seed)      
        random.seed

        self.toolbox = base.Toolbox()

        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", self.generate, size=self.problema.numJobs*self.problema.numMchs*2, pmin=0, pmax=self.problema.speed, smin=-2, smax=2)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", self.updateParticle, phi1=1.0, phi2=1.0)
        self.toolbox.register("evaluate", self.fitness)

    def generate(self, size, pmin, pmax, smin, smax):
        part = creator.Particle(np.random.uniform(pmin, pmax) for _ in range(size))
        part.speed = [np.random.uniform(smin, smax) for _ in range(size)]
        part.smin  = smin
        part.smax  = smax
        return part

    def updateParticle(self,iter, part, best, phi1, phi2):
        u1 = np.random.uniform(0, phi1)
        u2 = np.random.uniform(0, phi2)

        v_u1 = map(lambda x: self.c1[min(iter, self.iterations-1)]*u1*x, map(operator.sub, part.best, part))
        v_u2 = map(lambda x: self.c2[min(iter, self.iterations-1)]*u2*x, map(operator.sub, best, part))


        part.speed = list(map(operator.add, map(lambda x: x*self.w[min(iter, self.iterations-1)],part.speed), map(operator.add, v_u1, v_u2)))

        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)

        part[:] = list(map(operator.add, part, part.speed))

    def get_permutation(self, particle):
        permutation = []

        job   = particle[::2]
        speed = particle[1::2]

        job_index_sorted = sorted(enumerate(zip(job, speed)), key=lambda x: x[1])
        for i, (job, speed )in job_index_sorted:
            permutation.append(i // self.problema.numMchs)
            permutation.append(min(max(0,round(speed)), self.problema.speed - 1))
        return permutation

    def fitness(self, particle):
        return self.problema.objective_function_solution(self.get_permutation(particle))[0],
    
    def solve(self, path, timeout=None, replace=True, guardar=True):
        if not replace and os.path.exists(f'{path}-pso.json'):
            return     
        # Generar población aleatoria
        iteraciones          = 0                    # Número de iteraciones realizadas
        tiempo_inicial       = time.time()          # Tiempo en el que empieza
        tiempo_transcurrido  = 0                    # Tiempo total transcurrido desde el inicio de la resolucion
        tiempo_por_iteracion = 0                    # Tiempo medio consumido por iteracion
        stats = {}

        if not timeout:
            timeout = 60
        elif type(timeout) != float and type(timeout) != int:
            timeout = timeout[0][self.problema.numJobs-1] + timeout[0][self.problema.numMchs-1] + timeout[1][self.problema.rddd] + timeout[2][self.problema.speed-1]
            timeout = timeout / 1000

        pop = self.toolbox.population(n=self.num_particles)
        best = None
        # Mientras no se cumpla la condición
        while(tiempo_transcurrido + tiempo_por_iteracion < timeout):# and iteraciones < self.iterations):
            # Para cada partícula 
            # Evalua fitness global
            # Actualizar mejor posición
            for part in pop:
                part.fitness.values = self.toolbox.evaluate(part)
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
            # Recalcular la velocidad y actualizar posición 
            for part in pop:
                self.toolbox.update(iteraciones, part, best)                
            # print(best.fitness.values[0])
            # print(self.get_permutation(best))        
            # Actualización de las variables de control del tiempo de ejecución
            iteraciones += 1
            tiempo_transcurrido = time.time() - tiempo_inicial
            tiempo_por_iteracion = tiempo_transcurrido / iteraciones 

        if guardar:
            makespan, energy, tardiness   = self.problema.objective_function_solution(self.get_permutation(best))[1]
            stats["bestSolution"]         = self.get_permutation(best)
            stats["makespan"]             = int(makespan)
            stats["normalized_makespan"]  = self.problema.norm_makespan(makespan)
            stats["energy"]               = int(energy)
            stats["normalized_energy"]    = self.problema.norm_energy(energy)
            stats["tardiness"]            = int(tardiness)
            stats["normalized_tardiness"] = self.problema.norm_tardiness(tardiness)
            stats["bestObjectiveFunction"]= best.fitness.values[0]
            stats["tiempo_por_iteracion"] = tiempo_por_iteracion

            with open(f'{path}-pso.json','w') as file:
                json.dump(stats, file, indent=4)
        
        # del creator.Particle
        # del creator.Fitness

        return best.fitness.values[0]

class GRASP:   
    def __init__(self, dir, iterations, alpha, maxNeighbord, seed=0):
        self.iterations     = iterations
        self.alpha          = alpha
        self.maxNeighbord   = maxNeighbord

        with open(f'{dir}{dir.split("/")[-2]}','rb') as file:
            self.problema:JSP = pickle.load(file)

        # Se fija la semilla
        np.random.seed(seed)
        random.seed(seed)

    def construction(self):
        schedule = array.array('i',[])

        # Datos que se necesitan para calcular el makespan de manera eficiente
        orders_done              = [0] * self.problema.numJobs
        available_time_machines  = [0] * self.problema.numMchs
        end_time_last_operations = [0] * self.problema.numJobs
        tproc                    = [0] * self.problema.numJobs

        makespan, energy, tardiness = 0,0,0
        fitnessGlobal     = 0
        for _ in range(self.problema.numJobs*self.problema.numMchs):
            next_solutions = []
            job_candidates = [x for x in range(self.problema.numJobs) if orders_done[x] < self.problema.numMchs]
            for candidate, speed in itertools.product(job_candidates, range(self.problema.speed)):
                fitness,_,_,_ = self.problema.evalua_añadir_operacion(candidate, speed, makespan, energy, tardiness, orders_done, available_time_machines, end_time_last_operations, tproc, False)
                next_solutions.append((fitness, candidate, speed))

            next_solutions = sorted(next_solutions)
            threshold = next_solutions[0][0] + self.alpha*(next_solutions[-1][0] - next_solutions[0][0])
            rcl = []
            for i in range(len(next_solutions)):
                if next_solutions[i][0] > threshold:
                    break
                rcl.append(next_solutions[i])
            
            index    = np.random.choice(range(len(rcl)))
            _, candidate, speed = rcl[index]
            fitnessGlobal, makespan, energy, tardiness = self.problema.evalua_añadir_operacion(candidate, speed, makespan, energy, tardiness, orders_done, available_time_machines, end_time_last_operations, tproc, True)
            
            schedule.append(candidate)
            schedule.append(speed)
        
        return schedule,fitnessGlobal

    def improve(self, solution, fitness):
        # Primera fase del local search es un local search con cambio de posición de las operaciones
        best        = solution
        bestFitness = fitness
        for neighbor in self.neighborhood(solution):
            neighbor_fitness = self.problema.objective_function_solution(neighbor)[0]
            if neighbor_fitness < fitness:
                best        = neighbor
                bestFitness = neighbor_fitness
        return best, bestFitness

    def neighborhood(self, solution):
        a = list(itertools.combinations(range(0,len(solution),2),2))
        b = np.random.choice(range(len(a)), min(max(0, int(1e-5*len(a))),self.maxNeighbord))
        c  = [a[i] for i in b]
        for j1, j2 in c:
            if solution[j1] == solution[j2]:
                continue
            for s1, s2 in itertools.product(range(self.problema.speed),range(self.problema.speed)):
                new_solution = array.array('i',solution)
                new_solution[j1]     = solution[j2]
                new_solution[j1 + 1] = s1
                new_solution[j2]     = solution[j1]
                new_solution[j2 + 1] = s2
            yield new_solution

    def solve(self, path, timeout=None, replace=True, guardar=True):
        if not replace and os.path.exists(f'{path}-grasp.json'):
            return     
        # Generar población aleatoria
        iteraciones          = 0                    # Número de iteraciones realizadas
        tiempo_inicial       = time.time()          # Tiempo en el que empieza
        tiempo_transcurrido  = 0                    # Tiempo total transcurrido desde el inicio de la resolucion
        tiempo_por_iteracion = 0                    # Tiempo medio consumido por iteracion
        stats = {}

        if not timeout:
            timeout = 60
        elif type(timeout) == list:
            timeout = timeout[0][self.problema.numJobs-1] + timeout[0][self.problema.numMchs-1] + timeout[1][self.problema.rddd] + timeout[2][self.problema.speed-1]
            timeout = timeout / 1000

        best, bestFitness = None, np.inf
        # Mientras no se cumpla la condición
        while(tiempo_transcurrido + tiempo_por_iteracion < timeout):# and iteraciones < self.iterations):
            # Construction Phase
            solution, fitness = self.construction()
            # Local Search Phase
            # solution, fitness = self.improve(solution, fitness)
            # # Update Phase
            if fitness < bestFitness:
                best        = solution
                bestFitness = fitness
                #print(bestFitness)
            # Actualización de las variables de control del tiempo de ejecución
            iteraciones += 1
            tiempo_transcurrido = time.time() - tiempo_inicial
            tiempo_por_iteracion = tiempo_transcurrido / iteraciones


        (makespan, energy, tardiness)    = self.problema.objective_function_solution(best)[1]
        stats["bestSolution"]          = list(best)
        stats["makespan"]              = int(makespan)
        stats["normalized_makespan"]   = self.problema.norm_makespan(makespan)
        stats["energy"]                = int(energy)        
        stats["normalized_energy"]     = self.problema.norm_energy(energy)
        stats["tardiness"]             = int(tardiness)
        stats["normalized_tardiness"]  = self.problema.norm_tardiness(tardiness)
        stats["bestObjectiveFunction"] = bestFitness
        stats["tiempo_por_iteracion"]  = tiempo_por_iteracion

        with open(f'{path}-grasp.json','w') as file:
            json.dump(stats, file, indent=4)
        return bestFitness

class Gurobi:
    def __init__(self, dir, seed=0) -> None:
        self.dir = dir

    def solve(self, path, timeout=60, replace=True, guardar=True):
        # Recorrer todos los archivos .dzn en el directorio y subdirectorios
        for root, _, files in os.walk(self.dir):
            dzn_files = [file for file in files if file.endswith(".dzn")]
            print("DINK DONK LOS DZN FILES SON ESTOS", dzn_files)
            
            # Procesar cada archivo .dzn encontrado
            for dzn in dzn_files:
                # Generar el nombre del archivo JSON de salida basado en el nombre del archivo .dzn
                output_json_path = os.path.join(root, f"{os.path.splitext(dzn)[0]}-gurobi.json")
                print("El json lo estoy guardando como ", output_json_path)

                if not replace and os.path.exists(output_json_path):
                    continue
                
                with open(os.path.join(root, dzn), "r+", encoding="utf-8") as file:
                    lineas = file.readlines()
                    self.numJobs = int(lineas[1].split(".")[-1].split(";")[0])
                    self.numMchs = int(lineas[2].split(".")[-1].split(";")[0])
                    self.rddd = int(dzn.split('-')[1])
                    self.speed = int(dzn.split('-')[2].split('.')[0])
                
                model = Model(f"Minizinc/Models/RD/JSP{self.rddd}.mzn")
                model.add_file(os.path.join(root, dzn), parse_data=True)
                model.add_file(f"Solvers/MPCs/gurobi.mpc")
                
                self.instance = Instance(solver=Solver.lookup('gurobi'), model=model)

                if not timeout:
                    timeout = 60
                    print("Este es el primer timeout")
                else:
                    timeout = 30
                    print("Este es el segundo timeout")
                
                try:
                    result = self.instance.solve(timeout=timedelta(seconds=timeout))
                    print("Instancia creada, ahora voy a pasar a resolver")

                    if result:
                        save = result.__dict__
                        if "status" in save.keys(): save['status'] = str(result.status)
                        if "solution" in save.keys(): save["solution"] = result.solution.__dict__
                        if "time" in save["statistics"].keys(): save["statistics"]["time"] = save["statistics"]["time"].total_seconds() * 1000
                        if "optTime" in save["statistics"].keys(): save["statistics"]["optTime"] = save["statistics"]["optTime"].total_seconds() * 1000
                        if "flatTime" in save["statistics"].keys(): save["statistics"]["flatTime"] = save["statistics"]["flatTime"].total_seconds() * 1000
                        if "initTime" in save["statistics"].keys(): save["statistics"]["initTime"] = save["statistics"]["initTime"].total_seconds() * 1000
                        if "solveTime" in save["statistics"].keys(): save["statistics"]["solveTime"] = save["statistics"]["solveTime"].total_seconds() * 1000
                        print("He terminado de resolver la instancia")
                    else:
                        save = {'solution': 'None'}
                except Exception as e:
                    print(f"Ocurrió un error: {e}")
                    save = {'solution': 'None'}
                
                if guardar:
                    with open(output_json_path, 'w') as f:
                        json.dump(save, f, indent=4)
                else:
                    return save
        
class Cplex:
    def __init__(self, dir, seed=0) -> None:
        self.dir = dir

    def solve(self, path, timeout=60, replace=True, guardar=True):
        # Recorrer todos los archivos .dzn en el directorio y subdirectorios
        for root, _, files in os.walk(self.dir):
            dzn_files = [file for file in files if file.endswith(".dzn")]
            print("DINK DONK LOS DZN FILES SON ESTOS", dzn_files)
            
            # Procesar cada archivo .dzn encontrado
            for dzn in dzn_files:
                # Generar el nombre del archivo JSON de salida basado en el nombre del archivo .dzn
                output_json_path = os.path.join(root, f"{os.path.splitext(dzn)[0]}-cplex.json")
                print("El json lo estoy guardando como ", output_json_path)

                if not replace and os.path.exists(output_json_path):
                    continue
                
                with open(os.path.join(root, dzn), "r+", encoding="utf-8") as file:
                    lineas = file.readlines()
                    self.numJobs = int(lineas[1].split(".")[-1].split(";")[0])
                    self.numMchs = int(lineas[2].split(".")[-1].split(";")[0])
                    self.rddd = int(dzn.split('-')[1])
                    self.speed = int(dzn.split('-')[2].split('.')[0])
                
                model = Model(f"Minizinc/Models/RD/JSP{self.rddd}.mzn")
                model.add_file(os.path.join(root, dzn), parse_data=True)
                model.add_file(f"Solvers/MPCs/cplex.mpc")
                
                self.instance = Instance(solver=Solver.lookup('cplex'), model=model)

                if not timeout:
                    timeout = 60
                    print("Este es el primer timeout")
                else:
                    timeout = 30
                    print("Este es el segundo timeout")
                
                try:
                    result = self.instance.solve(timeout=timedelta(seconds=timeout))
                    print("Instancia creada, ahora voy a pasar a resolver")

                    if result:
                        save = result.__dict__
                        if "status" in save.keys(): save['status'] = str(result.status)
                        if "solution" in save.keys(): save["solution"] = result.solution.__dict__
                        if "time" in save["statistics"].keys(): save["statistics"]["time"] = save["statistics"]["time"].total_seconds() * 1000
                        if "optTime" in save["statistics"].keys(): save["statistics"]["optTime"] = save["statistics"]["optTime"].total_seconds() * 1000
                        if "flatTime" in save["statistics"].keys(): save["statistics"]["flatTime"] = save["statistics"]["flatTime"].total_seconds() * 1000
                        if "initTime" in save["statistics"].keys(): save["statistics"]["initTime"] = save["statistics"]["initTime"].total_seconds() * 1000
                        if "solveTime" in save["statistics"].keys(): save["statistics"]["solveTime"] = save["statistics"]["solveTime"].total_seconds() * 1000
                        print("He terminado de resolver la instancia")
                    else:
                        save = {'solution': 'None'}
                except Exception as e:
                    print(f"Ocurrió un error: {e}")
                    save = {'solution': 'None'}
                
                if guardar:
                    with open(output_json_path, 'w') as f:
                        json.dump(save, f, indent=4)
                else:
                    return save

# class Gecode:
#     def __init__(self, dir, seed=0) -> None:
#         self.dir = dir

#     def solve(self, path, timeout=60, replace=True, guardar=True):
#         if not replace and os.path.exists(f'{path}-gecode.json'):
#             return
        
#         dzn = [file for file in os.listdir(f"{self.dir}") if file[-3:] == "dzn"][0]
#         with open(f"{self.dir}/{dzn}","r+",encoding="utf-8") as file:
#             lineas = file.readlines()
#             self.numJobs = int(lineas[1].split(".")[-1].split(";")[0])
#             self.numMchs = int(lineas[2].split(".")[-1].split(";")[0])
#             self.rddd = int(dzn.split('-')[1])
#             self.speed = int(dzn.split('-')[2].split('.')[0])
#         model = Model(f"Minizinc/Models/RD/JSP{self.rddd}.mzn")
#         model.add_file(f"{self.dir}/{dzn}", parse_data=True)
#         model.add_file(f"Solvers/MPCs/gecode.mpc")
#         solver = Solver.lookup('gecode')
#         self.instance = Instance(solver= solver,model=model)
#         print(f"Solver configurado: {solver.name}")

#         if not timeout:
#             timeout = 60
#         else:
#             timeout = 500
#         try:
#             result = self.instance.solve(timeout=timedelta(seconds=timeout))
#             if result:
#                 save = result.__dict__
#                 if "status" in save.keys() : save['status'] = str(result.status)
#                 if "solution" in save.keys() : save["solution"] =  result.solution.__dict__
#                 if "time" in save["statistics"].keys() : save["statistics"]["time"] = save["statistics"]["time"].total_seconds() * 1000
#                 if "optTime" in save["statistics"].keys() : save["statistics"]["optTime"] = save["statistics"]["optTime"].total_seconds() * 1000
#                 if "flatTime" in save["statistics"].keys(): save["statistics"]["flatTime"] = save["statistics"]["flatTime"].total_seconds() * 1000
#                 if "initTime" in save["statistics"].keys()  : save["statistics"]["initTime"] = save["statistics"]["initTime"].total_seconds() * 1000
#                 if "solveTime" in save["statistics"].keys() : save["statistics"]["solveTime"] = save["statistics"]["solveTime"].total_seconds() * 1000
#             else:
#                 save = {'solution':'None'}
#         except Exception as e:
#             save = {'solution': 'None'}
#         if guardar:
#             with open(f'{path}-gecode.json', 'w') as f:
#                 json.dump(save, f, indent=4)
#         else:
#             return save

class Gecode:
    def __init__(self, dir, seed=0) -> None:
        self.dir = dir

    def solve(self, path, timeout=60, replace=True, guardar=True):
        # Recorrer todos los archivos .dzn en el directorio y subdirectorios
        for root, _, files in os.walk(self.dir):
            dzn_files = [file for file in files if file.endswith(".dzn")]
            print("DINK DONK LOS DZN FILES SON ESTOS", dzn_files)
            
            # Procesar cada archivo .dzn encontrado
            for dzn in dzn_files:
                # Generar el nombre del archivo JSON de salida basado en el nombre del archivo .dzn
                output_json_path = os.path.join(root, f"{os.path.splitext(dzn)[0]}-gecode.json")
                print("El json lo estoy guardando como ", output_json_path)

                if not replace and os.path.exists(output_json_path):
                    continue
                
                with open(os.path.join(root, dzn), "r+", encoding="utf-8") as file:
                    lineas = file.readlines()
                    self.numJobs = int(lineas[1].split(".")[-1].split(";")[0])
                    self.numMchs = int(lineas[2].split(".")[-1].split(";")[0])
                    self.rddd = int(dzn.split('-')[1])
                    self.speed = int(dzn.split('-')[2].split('.')[0])
                
                model = Model(f"Minizinc/Models/RD/JSP{self.rddd}.mzn")
                model.add_file(os.path.join(root, dzn), parse_data=True)
                model.add_file(f"Solvers/MPCs/gecode.mpc")
                
                self.instance = Instance(solver=Solver.lookup('gecode'), model=model)

                if not timeout:
                    timeout = 60
                    print("Este es el primer timeout")
                else:
                    timeout = 3
                    print("Este es el segundo timeout")
                
                try:
                    result = self.instance.solve(timeout=timedelta(seconds=timeout))
                    print("Instancia creada, ahora voy a pasar a resolver")

                    if result:
                        save = result.__dict__
                        if "status" in save.keys(): save['status'] = str(result.status)
                        if "solution" in save.keys(): save["solution"] = result.solution.__dict__
                        if "time" in save["statistics"].keys(): save["statistics"]["time"] = save["statistics"]["time"].total_seconds() * 1000
                        if "optTime" in save["statistics"].keys(): save["statistics"]["optTime"] = save["statistics"]["optTime"].total_seconds() * 1000
                        if "flatTime" in save["statistics"].keys(): save["statistics"]["flatTime"] = save["statistics"]["flatTime"].total_seconds() * 1000
                        if "initTime" in save["statistics"].keys(): save["statistics"]["initTime"] = save["statistics"]["initTime"].total_seconds() * 1000
                        if "solveTime" in save["statistics"].keys(): save["statistics"]["solveTime"] = save["statistics"]["solveTime"].total_seconds() * 1000
                        print("He terminado de resolver la instancia")
                    else:
                        save = {'solution': 'None'}
                except Exception as e:
                    print(f"Ocurrió un error: {e}")
                    save = {'solution': 'None'}
                
                if guardar:
                    with open(output_json_path, 'w') as f:
                        json.dump(save, f, indent=4)
                else:
                    return save
class ORTOOLS:
    def __init__(self, dir, seed=0) -> None:
        self.dir = dir
 
    def solve(self, path, timeout=60, replace=True, guardar=True):
        if not replace and os.path.exists(f'{path}-ortools.json'):
            return
        
        dzn = [file for file in os.listdir(f"{self.dir}") if file[-3:] == "dzn"][0]
        with open(f"{self.dir}/{dzn}","r+",encoding="utf-8") as file:
            lineas = file.readlines()
            self.numJobs = int(lineas[1].split(".")[-1].split(";")[0])
            self.numMchs = int(lineas[2].split(".")[-1].split(";")[0])
            self.rddd = int(dzn.split('-')[1])
            self.speed = int(dzn.split('-')[2].split('.')[0])
        model = Model(f"Minizinc/Models/RD/JSP{self.rddd}.mzn")
        model.add_file(f"{self.dir}/{dzn}", parse_data=True)
        model.add_file(f"Solvers/MPCs/cpsatlp.mpc")
        self.instance = Instance(solver=Solver.lookup('cpsatlp'), model=model)
    
        if not timeout:
            timeout = 60
        else:
            timeout = 3
        try:
            result = self.instance.solve(timeout=timedelta(seconds=timeout))
            if result:
                save = result.__dict__
                if "status" in save.keys() : save['status'] = str(result.status)
                if "solution" in save.keys() : save["solution"] =  result.solution.__dict__
                if "time" in save["statistics"].keys() : save["statistics"]["time"] = save["statistics"]["time"].total_seconds() * 1000
                if "optTime" in save["statistics"].keys() : save["statistics"]["optTime"] = save["statistics"]["optTime"].total_seconds() * 1000
                if "flatTime" in save["statistics"].keys(): save["statistics"]["flatTime"] = save["statistics"]["flatTime"].total_seconds() * 1000
                if "initTime" in save["statistics"].keys()  : save["statistics"]["initTime"] = save["statistics"]["initTime"].total_seconds() * 1000
                if "solveTime" in save["statistics"].keys() : save["statistics"]["solveTime"] = save["statistics"]["solveTime"].total_seconds() * 1000
            else:
                save = {'solution':'None1'}
        except Exception as e:
            save = {'solution': f'None2 - Error: {str(e)}'}
        if guardar:  
            with open(f'{path}-ortools.json', 'w') as f:
                json.dump(save, f, indent=4)
        else:
            return save
        

def TimeLimitIntance(min_time:int, max_time:int, num: list, power:int=2):
    start = np.power(min_time//len(num), 1/float(2))
    stop = np.power(max_time//len(num), 1/float(2))
    res = [np.power(np.linspace(start, stop, num=n), 2) for n in num]
    # TimeLimitIntance(200,1*60*1000,[100,3,5]) 
    return  res

if __name__ == "__main__":
    POPULATION_SIZE = 1000
    ITERATIONS      = 100
    DIR             = r"./ConjuntoTFG-dzn/Problema100/"
    INDIVIDUO       = r"./ConjuntoTFG-dzn/Problema100/Problema100"
    MUTATION_RATE   = 0.2
    CROSS_RATE      = 1.0

    # Inicialización de los parámetros de la libreria DEAP
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode="i", fitness=creator.Fitness)
    creator.create("IndividualFloat", array.array, typecode="d", fitness=creator.Fitness)
    creator.create("Particle", list, fitness=creator.Fitness, speed=list, smin=None, smax=None, best=None)

    # Cargamos el solver que se quiere utilizar
    solvers = [(GA, [DIR,POPULATION_SIZE, ITERATIONS, CROSS_RATE, MUTATION_RATE,90], 'Algoritmo Genético'),
              (PSO, [DIR, POPULATION_SIZE, ITERATIONS, [0.7]*ITERATIONS, [0.1]*ITERATIONS,np.linspace(0.7, 0.3, ITERATIONS),10], 'PSO')]
              #(GRASP, [DIR, ITERATIONS, 0.1, 100, 10], 'GRASP')]
    for solver, params, name in solvers:
        a = time.time()
        b = solver(*params)
        print(name, "\t: ", b.solve(INDIVIDUO, timeout=60))
        print(time.time()- a)