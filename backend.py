import random
import math
from flask import Flask, render_template

class GeneticAlgorithmTracker:
    def __init__(self):
        self.generations_data = []
        self.population_size = 100
        self.number_of_generations = 10
        self.enemy_composition = [500, 0, 0]
        self.total_enemy_units = sum(self.enemy_composition)
        self.unit_costs = {
            "infantry": 10, 
            "cavalry": 20, 
            "archers": 15
        }
        self.unit_win_probability = {
            "infantry": {"infantry": 0.5, "cavalry": 0.67, "archers": 0.33},
            "cavalry": {"infantry": 0.33, "cavalry": 0.5, "archers": 0.67},
            "archers": {"infantry": 0.67, "cavalry": 0.33, "archers": 0.5},
        }

    #Initialization function
    def init_population(self):
        population = []
        for _ in range(self.population_size):
            gene1 = random.randint(0, self.total_enemy_units)
            gene2 = random.randint(0, self.total_enemy_units - gene1)
            gene3 = self.total_enemy_units - (gene1 + gene2)
            individual = [gene1, gene2, gene3]
            random.shuffle(individual)
            population.append(individual)
        return population

    #Selection function
    def evaluate_fitness_score(self, composition, enemy_composition, unit_win_probability, unit_costs):
        infantry, cavalry, archers = composition
        enemy_infantry, enemy_cavalry, enemy_archers = enemy_composition
        total_units = infantry + cavalry + archers
        enemy_total_units = enemy_infantry + enemy_cavalry + enemy_archers
        total_win_probability = 0

        number_advantage_weigh = 0.33
        cost_penalty_weigh = 0.5

        # Calculates number advantage
        troop_ratio = total_units / enemy_total_units
        number_advantage = 1 + (troop_ratio - 1) * number_advantage_weigh

        # Calculates unit type advantage
        total_win_probability += infantry * (
            (enemy_infantry * unit_win_probability["infantry"]["infantry"]) +
            (enemy_cavalry * unit_win_probability["infantry"]["cavalry"]) +
            (enemy_archers * unit_win_probability["infantry"]["archers"])
        )
        total_win_probability += cavalry * (
            (enemy_infantry * unit_win_probability["cavalry"]["infantry"]) +
            (enemy_cavalry * unit_win_probability["cavalry"]["cavalry"]) +
            (enemy_archers * unit_win_probability["cavalry"]["archers"])
        )
        total_win_probability += archers * (
            (enemy_infantry * unit_win_probability["archers"]["infantry"]) +
            (enemy_cavalry * unit_win_probability["archers"]["cavalry"]) +
            (enemy_archers * unit_win_probability["archers"]["archers"])
        )
        
        # Normalize 
        total_possible_battles = (infantry + cavalry + archers) * (enemy_infantry + enemy_cavalry + enemy_archers)
        if total_possible_battles > 0:
            total_win_probability = (total_win_probability * number_advantage) / total_possible_battles
        else:
            total_win_probability = 0

        # Calculates overall fitness with cost penalty
        cost_penalty = 0
        fitness = total_win_probability
        composition_cost = (infantry * self.unit_costs["infantry"] +
                            cavalry * self.unit_costs["cavalry"] +
                            archers * self.unit_costs["archers"]
                            )
        enemy_composition_cost = (enemy_infantry * self.unit_costs["infantry"] +
                        enemy_cavalry * self.unit_costs["cavalry"] +
                        enemy_archers * self.unit_costs["archers"]
                        )
        if composition_cost > enemy_composition_cost:
            cost_penalty = ((composition_cost - enemy_composition_cost) / enemy_composition_cost) * cost_penalty_weigh
            fitness = total_win_probability * (1 - cost_penalty)

        return fitness, total_win_probability, cost_penalty

    #Crossover function
    def crossover_uniform(self, parent1, parent2):
        offspring1 = []
        offspring2 = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                offspring1.append(parent1[i])
                offspring2.append(parent2[i])
            else:
                offspring1.append(parent2[i])
                offspring2.append(parent1[i])
        return offspring1, offspring2

    #Selection function
    def tournament_selection(self, population, fitness_scores):
        tournament_size = 5
        selected = []
        for i in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            winner_index = tournament_indices[0]
            for j in tournament_indices[1:]:
                if fitness_scores[j] > fitness_scores[winner_index]:
                    winner_index = j
            selected.append(population[winner_index])
        return selected

    #Mutation function
    def integer_mutate(self, individual, mutation_rate):
        mutated_individual = list(individual)
        
        for i in range(len(mutated_individual)):
            if random.random() < mutation_rate:
                change = random.randint(-math.ceil(abs(mutated_individual[i])), math.ceil(abs(mutated_individual[i]))) if mutated_individual[i] != 0 else 0
                mutated_individual[i] = max(0, mutated_individual[i] + change)

        return mutated_individual

    def run_genetic_algorithm(self, mutation_rate=0.1):
        population = self.init_population()
        best_individual = None
        best_fitness = -1

        for generation in range(self.number_of_generations):
            
            # Fitness evaluation
            fitness_scores = []
            total_win_probability_scores = []
            cost_penalty_scores = []
            
            for individual in population:
                fitness_score, total_win_probability_score, cost_penalty_score = self.evaluate_fitness_score(
                    individual, self.enemy_composition, self.unit_win_probability, self.unit_costs
                )
                fitness_scores.append(fitness_score)
                total_win_probability_scores.append(total_win_probability_score)
                cost_penalty_scores.append(cost_penalty_score)
            
            # Track generation best individual
            current_best_index = fitness_scores.index(max(fitness_scores))
            if fitness_scores[current_best_index] > best_fitness:
                best_fitness = fitness_scores[current_best_index]
                best_individual = population[current_best_index]
                best_win_probability = total_win_probability_scores[current_best_index]
                best_cost_penalty = cost_penalty_scores[current_best_index]

            # Store generation data
            generation_data = {
                'enemy_composition': self.enemy_composition,
                'generation_number': generation + 1,
                'population': population,
                'fitness_scores': fitness_scores,
                'total_win_probability_scores': total_win_probability_scores,
                'cost_penalty_scores': cost_penalty_scores,
                'best_individual': best_individual,
                'best_fitness': best_fitness
            }
            self.generations_data.append(generation_data)

            # Selection
            parents = self.tournament_selection(population, fitness_scores)

            # Crossover
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                offspring1, offspring2 = self.crossover_uniform(parent1, parent2)
                new_population.extend([offspring1, offspring2])

            # Mutation
            population = [
                self.integer_mutate(individual, mutation_rate) 
                for individual in new_population
            ]

        return self.generations_data

# Flask App
app = Flask(__name__)

@app.route('/')
def index():
    ga_tracker = GeneticAlgorithmTracker()
    generations_data = ga_tracker.run_genetic_algorithm()
    return render_template('frontend.html', generations_data=generations_data)

if __name__ == '__main__':
    app.run(debug=True)