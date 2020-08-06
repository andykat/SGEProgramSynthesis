import random
import copy
import math
from enum import Enum
import networkx as nx
from collections import defaultdict
from multiprocessing import Pool
import os
import json
import signal
import numpy as np


# timeout functions used to exit of execution of sythensized code when the synthesized code requires too many
# calculations to finish running in reasonable time
class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException
    # print("timeout exception")
    # pass


class SGE:

    def __init__(self, file_name):
        self.file_name = file_name
        self.rules = {}
        self.non_terminals = []
        random.seed()
        self.population_size = 100  # overwritten in main
        self.genotype_max = 99999999  # maximum value for each integer
        self.genotype_min = 0  # minimum value for each integer

        self.tournament_k = 7  # number of entrants for each tournament
        self.tournament_p = 0.7  # probability of selecting winner
        self.top_performing_carry = 3  # number of top performing sequences from previous population carried over

        self.gene_mutation_chance = 0.15  # chance that a gene mutates
        self.average_best_fitness = []
        self.average_best_fitness_N = 5  # calculation of average top fitness to print out

        self.recursion_max = 2  # level of recursion, overwritten in main

        self.helper_code = ""
        self.test_string = ""
        self.train_string = ""
        
        self.final_nonterminal = ""  # top level nonterminal

        self.currentPopulation = []  # list of genotypes for the current generation
        self.fitness = []  # fitness calculation of current generation
        self.fitness_indicies = []  # match fitnesses with genotypes after sorting
        self.highest_performers = []
        self.highest_fitnesses = []
        self.population_nonterminal_count = []
        self.newpopulation = []

        self.fitness_cache = {}  # stores phenotypes and their fitness to speed up computation
        self.cache_use_per_iteration = []
        self.iteration_cache_use_count = 0

        self.grammar_replacement = {"greater_equal_than": ">=",
                                    "less_equal_than": "<=", "loop_break_constant": "10",
                                    "greater_than": ">", "less_than": "<"}

        # variables used to minimize cycles in phenotype translation
        self.graph = nx.Graph()
        self.nonterminals_in_cycle = []

        self.is_referenced_by = {}
        self.possible_rhs_for_endcycle = {}
        self.not_possible_rhs_for_endcycle = {}
        self.nonterminal_terminates = set()
        self.secondary_nonterminal_terminates = set()
        self.count_iteration = 0
        signal.signal(signal.SIGALRM, timeout_handler)

        # variables for keeping track the rate in which the synthesized code has an execution error
        self.error_counts = 0
        self.error_programs_total = 0
        self.error_save = []

    def read_supplementary_files(self):
        """
        reads helper code, and testing and training datasets
        :return:
        """
        f_name = "../helper_codes/" + self.file_name + "_Helper.txt"
        with open(f_name) as fp:
            self.helper_code = fp.read()

        f_name = "../datasets/" + self.file_name + "_Test.txt"
        with open(f_name) as fp:
            self.test_string = fp.read()

        f_name = "../datasets/" + self.file_name + "_Train.txt"
        with open(f_name) as fp:
            self.train_string = fp.read()

        self.helper_code = self.helper_code.replace("<train>", self.train_string.replace("\n", "\n  "))

    def read_bnf_file(self):
        """
        Reads the grammar file and converts it into a grammar
        Runs methods to calculate cycles and create a grammar that minimizes cycles
        :return:
        """
        equation_split = " ::= "
        f_name = "../grammars/" + self.file_name + ".bnf"
        count = 0
        for line in open(f_name, 'r'):
            # take left hand
            if line.startswith("<") and equation_split in line:
                t_split = line.split(equation_split)
                lhs = t_split[0].strip()
                # print(lhs)
                rhs = t_split[1].strip("\n")

                # transform rhs into orGroup of expressions
                or_separated_list = rhs.split("|")

                o = OrGroup()

                for production in or_separated_list:
                    expression = production.split("'")
                    s = Sentence()
                    for obj in expression:
                        if obj != "":
                            if "<" in obj:
                                obj_copy = obj[:]
                                while "<" in obj_copy:
                                    start_index = obj_copy.find("<")
                                    end_index = obj_copy.find(">")
                                    sub_string = obj_copy[start_index: end_index+1]
                                    obj_copy = obj_copy[end_index+1:]
                                    s.append_object(sub_string, TerminalType.NONTERMINAL)

                            else:
                                obj = obj.replace("\\n", "\n")
                                for replace_key in self.grammar_replacement:
                                    obj = obj.replace(replace_key, self.grammar_replacement[replace_key])

                                s.append_object(obj, TerminalType.TERMINAL)
                    o.expressions.append(s)

                self.rules[lhs] = Production(lhs, o)
                if count == 0:
                    self.final_nonterminal = lhs
                    count += 1

        self.process_grammar_cycles()

        # print the final grammar
        # for rule_lhs, rule in self.rules.items():
        #     print("lhs:" + rule_lhs)
        #
        #     for expression in rule.rhs.expressions:
        #         for ob in expression.objects:
        #             print(ob, end='')
        #         print("|", end='')
        #     print("\n")

        # for lhs, rule in self.rules.items():
        #     self.non_terminals.append(lhs)
        # print("\nnonterminals")
        # print(self.non_terminals)
        # print("\n")

    def process_grammar_cycles(self):
        """
        Gets the cycle data about the grammar
        Converts the grammar into an almost non-cyclical grammar
        :return:
        """
        self.find_grammar_cycle_end_points()
        self.create_grammar_graph()
        lhs_queue = [self.final_nonterminal]
        has_popped = []
        while len(lhs_queue) > 0:
            lhs = lhs_queue.pop(0)
            self.non_terminals.append(lhs[:])
            has_popped.append(lhs)
            or_group = self.rules[lhs].rhs
            for expression in or_group.expressions:
                for i in range(0, len(expression.objects)):
                    if expression.object_types[i] == TerminalType.NONTERMINAL:
                        t_nt = expression.objects[i]
                        if (t_nt not in has_popped) and (t_nt not in lhs_queue) and 'lvl' not in t_nt:
                            lhs_queue.append(t_nt)

            if lhs in self.nonterminals_in_cycle:
                # we only create levels of the nonterminal if the nonterminal is in a cycle
                for i in range(0, self.recursion_max):
                    will_add_rule_flag = True
                    or_group_deep_copy = copy.deepcopy(or_group)
                    new_or_group = copy.deepcopy(or_group)
                    new_lhs = copy.deepcopy(lhs)
                    # append lvl-1 to the lfh expression
                    if i > 0:
                        new_lhs = self.append_level_to_nonterminal(lhs, i - 1)

                    if i < self.recursion_max - 1:
                        for j in range(0, len(or_group_deep_copy.expressions)):
                            expression = or_group_deep_copy.expressions[j]
                            for k in range(0, len(expression.objects)):
                                if expression.object_types[k] == TerminalType.NONTERMINAL:
                                    # check is object is in a cycle
                                    if expression.objects[k] in self.nonterminals_in_cycle:
                                        nonterminal_lvl_string = \
                                            self.append_level_to_nonterminal(expression.objects[k], i)
                                        new_or_group.expressions[j].objects[k] = nonterminal_lvl_string

                    else:
                        # For the last level, we remove any expressions that include nonterminals part of a cycle
                        # if the or group is empty, then we do not add it
                        new_or_group = OrGroup()
                        for j in range(len(or_group_deep_copy.expressions)):
                            # check if this rhs can lead to a non cycle
                            leads_to_non_cycle = False
                            if j in self.possible_rhs_for_endcycle[lhs]:
                                leads_to_non_cycle = True
                            else:
                                leads_to_non_cycle = True
                                # check if one of the expressions has an expression that terminates
                                expression = or_group_deep_copy.expressions[j]
                                for k in range(0, len(expression.objects)):
                                    if expression.object_types[k] == TerminalType.NONTERMINAL:
                                        if expression.objects[k] not in self.nonterminal_terminates \
                                                or lhs == expression.objects[k]:
                                            leads_to_non_cycle = False
                                            break

                            if leads_to_non_cycle:
                                expression = copy.deepcopy(or_group_deep_copy.expressions[j])
                                for k in range(0, len(expression.objects)):
                                    if expression.object_types[k] == TerminalType.NONTERMINAL:
                                        # check is object is in a cycle
                                        if expression.objects[k] in self.nonterminals_in_cycle:
                                            nonterminal_lvl_string = self.append_level_to_nonterminal(
                                                expression.objects[k], i-1)  # maintain maximum recursion level
                                            expression.objects[k] = nonterminal_lvl_string

                                # add expression to or group
                                new_or_group.expressions.append(expression)

                    # if or group has no expressions, then nonterminal is never called
                    if len(new_or_group.expressions) == 0:
                        new_or_group.expressions.append(copy.deepcopy(self.rules['<blank>'].rhs.expressions[0]))

                    self.rules[new_lhs] = Production(new_lhs, new_or_group)
                    if new_lhs not in self.non_terminals:
                        self.non_terminals.append(new_lhs)

    def create_grammar_graph(self):
        """
        Calculates which nonterminals are in cycles
        :return:
        """
        for rule_lhs, rule in self.rules.items():
            for expression in rule.rhs.expressions:
                for i in range(len(expression.objects)):
                    if expression.object_types[i] == TerminalType.NONTERMINAL:
                        self.graph.add_edge(rule_lhs, expression.objects[i])

        cycle_basis = nx.cycle_basis(self.graph, self.final_nonterminal)
        for cycle in cycle_basis:
            for node in cycle:
                if node not in self.nonterminals_in_cycle:
                    self.nonterminals_in_cycle.append(node)

    def find_grammar_cycle_end_points(self):
        """
        Finds the paths through the grammar which will terminate so that the last level
        of recursion can end correctly
        :return:
        """
        intial_set = {self.final_nonterminal}
        # intial_set = [self.final_nonterminal]
        for rule_lhs, _ in self.rules.items():
            self.possible_rhs_for_endcycle[rule_lhs] = []
            self.not_possible_rhs_for_endcycle[rule_lhs] = []

        self.grammar_dps(self.final_nonterminal, intial_set)

        for lhs in self.possible_rhs_for_endcycle:
            if len(self.possible_rhs_for_endcycle[lhs]) > 0:
                self.nonterminal_terminates.add(lhs)

        # print("non cycle endings")
        # print(self.possible_rhs_for_endcycle)

    def grammar_dps(self, current_nt, nt_set):
        """
        dfs through the grammar to find noncyclical paths
        :param current_nt:
        :param nt_set:
        :return:
        """
        nt_terminates = False
        for i in range(len(self.rules[current_nt].rhs.expressions)):
            expression = self.rules[current_nt].rhs.expressions[i]
            expression_terminates = True
            for j in range(len(expression.objects)):
                if expression.object_types[j] == TerminalType.NONTERMINAL:
                    if expression.objects[j] not in nt_set:
                        nt_set.add(expression.objects[j])
                        did_complete = self.grammar_dps(expression.objects[j], nt_set)
                        if not did_complete:
                            expression_terminates = False
                        nt_set.remove(expression.objects[j])
                    else:
                        expression_terminates = False

            if expression_terminates:
                if i not in self.possible_rhs_for_endcycle[current_nt]:
                    if i not in self.not_possible_rhs_for_endcycle[current_nt]:
                        self.possible_rhs_for_endcycle[current_nt].append(i)
                nt_terminates = True
            else:
                if i not in self.not_possible_rhs_for_endcycle[current_nt]:
                    self.not_possible_rhs_for_endcycle[current_nt].append(i)
                    # remove impossible rhs from possible rhs
                    if i in self.possible_rhs_for_endcycle[current_nt]:
                        self.possible_rhs_for_endcycle[current_nt].remove(i)

        return nt_terminates

    @staticmethod
    def append_level_to_nonterminal(stri, level):
        """
        adds the 'level_n' to nonterminals to denote the level of recursion
        :param stri:
        :param level:
        :return:
        """
        position = stri.find('>')
        newstr = ""
        if position < 0:
            print("error: > not found in nonterminal")
        else:
            newstr = stri[:position] + "lvl" + str(level) + stri[position:]

        return newstr

    def initialize_population(self):
        """
        Initializes the first generation
        :return:
        """
        self.currentPopulation = []
        self.fitness = []
        for i in range(0, self.population_size):
            gene_dict = {}
            for key in self.non_terminals:
                genotype_max_length = 1
                gene_dict[key] = []
                for j in range(0, genotype_max_length):
                    gene_dict[key].append(self.random_genotype())
            self.currentPopulation.append(gene_dict)
            self.fitness.append(0)

    def random_genotype(self):
        """
        Random genotype between 0 and 99999999
        :return:
        """
        return random.randint(self.genotype_min, self.genotype_max)

    def mutate_parent(self, parent, nonterminal_count):
        """
        Mutates a genotype by going through each nonterminal and with probability gene_mutation_chance,
        changes one of the genes for that nonterminal
        :param parent:
        :param nonterminal_count:
        :return:
        """
        for nt in self.non_terminals:
            if random.uniform(0, 1) < self.gene_mutation_chance:
                if nonterminal_count[nt] > 0:
                    rand_index = random.randint(0, nonterminal_count[nt]-1)
                    parent[nt][rand_index] = self.random_genotype()
        return parent

    def recombination_cross(self, parents, nonterminal_counts):
        """
        Combines two parent genotypes into two children genotypes
        :param parents:
        :param nonterminal_counts:
        :return:
        """
        child = [{}, {}]
        mutated_parents = []
        for parent in parents:
            mutated_parents.append(copy.deepcopy(parent))
        # mutate genes
        for i in range(0, len(mutated_parents)):
            mutated_parents[i] = self.mutate_parent(mutated_parents[i], nonterminal_counts[i])

        for nt in self.non_terminals:
            rand_n = random.randint(0, 1)
            t_max = min(nonterminal_counts[0][nt], nonterminal_counts[1][nt])
            crossover_point = random.randint(0, t_max)
            other_n = rand_n + 1
            if other_n == 2:
                other_n = 0
            child[0][nt] = []
            child[1][nt] = []

            child[0][nt] = child[0][nt] + mutated_parents[rand_n][nt][:crossover_point]
            child[0][nt] = child[0][nt] + mutated_parents[other_n][nt][crossover_point:]
            child[1][nt] = child[1][nt] + mutated_parents[other_n][nt][:crossover_point]
            child[1][nt] = child[1][nt] + mutated_parents[rand_n][nt][crossover_point:]

        return child

    def translate_seq_to_phenotype(self, genes, remove_nonterminals=True):
        """
        Converts genotype into a phenotype
        :param genes:
        :param remove_nonterminals:
        :return:
        """
        cur_objects = []
        curobject_types = []
        cur_objects.append(self.final_nonterminal)
        curobject_types.append(TerminalType.NONTERMINAL)
        nonterminal_count = {}
        nonterminal_index_start = 0
        loop_break_count = 0
        for nt in self.non_terminals:
            nonterminal_count[nt] = 0
        while TerminalType.NONTERMINAL in curobject_types:
            # find next non terminal
            non_terminal_object = ""
            nonterminal_index = -1
            for i in range(nonterminal_index_start, len(curobject_types)):
                if curobject_types[i] == TerminalType.NONTERMINAL:
                    non_terminal_object = cur_objects[i]

                    nonterminal_index = i
                    break
                else:
                    nonterminal_index_start = i
            if nonterminal_index < 0:
                print("no terminalfound")
                break
            rule = self.rules[non_terminal_object]
            or_group = rule.rhs
            n_or_groups = len(or_group.expressions)

            # select an expression from or groups
            if nonterminal_count[non_terminal_object] >= len(genes[non_terminal_object]):
                # !! create new value
                genes[non_terminal_object].append(self.random_genotype())

            value = genes[non_terminal_object][nonterminal_count[non_terminal_object]]
            nonterminal_count[non_terminal_object] += 1
            index = value % n_or_groups
            expression = or_group.expressions[index]

            cur_objects.pop(nonterminal_index)
            curobject_types.pop(nonterminal_index)
            found_loop_break = False
            for i in range(0, len(expression.objects)):
                exp = expression.objects[i]
                if "loopBreak" in expression.objects[i]:
                    exp = exp.replace("loopBreak%", "loopBreak" + str(loop_break_count))
                    found_loop_break = True

                cur_objects.insert(nonterminal_index + i, exp)
                curobject_types.insert(nonterminal_index + i, expression.object_types[i])
            if found_loop_break:
                loop_break_count += 1
        if remove_nonterminals:
            i = 0
            while i < len(cur_objects):
                if curobject_types[i] == TerminalType.NONTERMINAL:
                    cur_objects.pop(i)
                    curobject_types.pop(i)
                    i -= 1
                i += 1
            return [[cur_objects, curobject_types], nonterminal_count]

        return [[cur_objects, curobject_types], nonterminal_count]

    # returns a tournament selected genotype
    def tournament_selection(self, k):
        """
        Uses tournament selection to select which genotype moves on. For a tournament size of k, we loop through
        the genotypes sorted by their fitness. Each genotype we loop through has a probability p of getting selected
        as the winner
        :param k:
        :return:
        """
        selected = []
        for i in range(0, k):
            selected.append(random.randint(0, self.population_size-1))
        selected = sorted(selected)

        for i in range(0, k):
            x = random.uniform(0, 1)
            if x < self.tournament_p:
                # return self.currentPopulation[self.fitness_indicies[selected[i]][0]]
                return self.fitness_indicies[selected[i]][0]
        # return self.currentPopulation[self.fitness_indicies[selected[0]][0]]
        return self.fitness_indicies[selected[0]][0]

    def create_children(self):
        """
        Selects random parents from tournament selection, and creates two children from them.
        :return:
        """
        parents = []
        parent0index = self.tournament_selection(self.tournament_k)
        parent1index = self.tournament_selection(self.tournament_k)
        parents.append(self.currentPopulation[parent0index])
        parents.append(self.currentPopulation[parent1index])
        nonterminal_counts = [self.population_nonterminal_count[parent0index],
                              self.population_nonterminal_count[parent1index]]

        children = self.recombination_cross(parents, nonterminal_counts)
        return children

    def run_iterations(self, iterations):
        """
        Main function
        Creates the population, and runs step() for each generation
        :param iterations:
        :return:
        """
        self.initialize_population()
        self.average_best_fitness = []
        self.cache_use_per_iteration = []

        # run the iterations
        iteration_count = 0
        success_flag = False
        for i in range(0, iterations):
            self.count_iteration = i
            if i % 25 == 0:
                print("iteration: " + str(i))
            iteration_count += 1
            success_flag = self.step()

            self.error_save.append(self.error_counts / self.error_programs_total)

            self.error_programs_total = 0
            self.error_counts = 0
            if success_flag:
                break

        # get highest performers
        for i in range(0, self.population_size):
            seq = self.currentPopulation[i]
            phen = self.translate_seq_to_phenotype(seq)[0][0]
            code = self.translate_objects_into_code(phen)
            self.fitness[i] = self.calculate_fitness(code, increase_cache_count=False)
            # get the top performing sequences

        self.fitness_indicies = []
        for i in range(0, self.population_size):
            self.fitness_indicies.append([i, self.fitness[i]])
        sorted(self.fitness_indicies, key=lambda x: x[1], reverse=True)
        self.highest_performers = []
        self.highest_fitnesses = []
        number_of_top_performers = 5

        for i in range(0, number_of_top_performers):
            phen = self.translate_seq_to_phenotype(self.currentPopulation[self.fitness_indicies[i][0]])[0][0]
            code = self.translate_objects_into_code(phen)
            tabbed_code = self.insert_tabs(code)
            # self.highest_performers.append(self.currentPopulation[self.fitness_indicies[i][0]])
            self.highest_performers.append(tabbed_code)
            self.highest_fitnesses.append(self.fitness_indicies[i][1])
        # print(self.highest_fitnesses[0])
        # print("error rate")
        # tstr = "Errors: " + str(self.error_counts)
        # print(tstr)
        # tstr = "Total Programs Run: " + str(self.error_programs_total)
        # print(tstr)
        iteration_index = []
        for i in range(len(self.error_save)):
            iteration_index.append(i+1)
        iterations = np.array(iteration_index)
        errors = np.array(self.error_save)
        file_name = self.file_name + "iterations"
        np.save(file_name, iterations)
        file_name = self.file_name + "errors"
        np.save(file_name, errors)

        abf = np.array(self.average_best_fitness)
        file_name = self.file_name + "fitness"
        np.save(file_name, abf)

        return [self.highest_performers, self.highest_fitnesses, success_flag, iteration_count,
                self.cache_use_per_iteration]

    # def fitness_subprocess(phen):
    #     code = self.translate_objects_into_code(phen)
    #     return self.calculate_fitness(code)

    def step(self):
        """
        Goes through one generation
        :return:
        """
        self.population_nonterminal_count = []
        self.iteration_cache_use_count = 0
        # print("getting fitness")
        # get the fitness of all sequences
        for i in range(0, self.population_size):
            # print(i)
            seq = self.currentPopulation[i]
            # print("translate seq to phen")
            result = self.translate_seq_to_phenotype(seq, remove_nonterminals=True)
            phen = result[0][0]
            nonterminal_count = result[1]
            self.population_nonterminal_count.append(nonterminal_count)
            # print("translate to code")
            code = self.translate_objects_into_code(phen)
            # self.fitness[i] = self.harmonicFitness(phen)
            # print("calculate fitness")
            self.fitness[i] = self.calculate_fitness(code)

        # get the top performing sequences

        self.fitness_indicies = []
        for i in range(0, self.population_size):
            self.fitness_indicies.append([i, self.fitness[i]])

        self.fitness_indicies = sorted(self.fitness_indicies, key=lambda x: x[1], reverse=True)

        afitness = 0.0
        for i in range(0, self.average_best_fitness_N):
            afitness += self.fitness_indicies[i][1]
        afitness /= -float(self.average_best_fitness_N)
        self.average_best_fitness.append(afitness)

        self.newpopulation = []

        # add the best performing sequence from last population
        for i in range(0, self.top_performing_carry):
            self.newpopulation.append(copy.deepcopy(self.currentPopulation[self.fitness_indicies[i][0]]))

        # fill in the rest of the new population with children
        while len(self.newpopulation) < self.population_size:
            # crossed children
            children = self.create_children()

            self.newpopulation.append(children[0])
            if len(self.newpopulation) < self.population_size:
                self.newpopulation.append(children[1])

        # print top performers
        # print("Top Performers:")
        #
        if self.count_iteration % 25 == 0:
            print("Top Average Fitness: %.2f" % afitness)

        flag = False
        if math.fabs(self.fitness_indicies[0][1]) < 0.0001:
            flag = True

        self.currentPopulation = self.newpopulation
        self.cache_use_per_iteration.append(self.iteration_cache_use_count)
        return flag

    def translate_objects_into_code(self, objects):
        """
        Formats the phenotype correctly into code
        :param objects:
        :return:
        """
        # print("translate_objects_into_code")
        # print(objects)
        code = ""
        for s in objects:
            code += s

        # return self.python_filter(code)
        return self.insert_tabs(code)

    def calculate_fitness(self, code, increase_cache_count=True):
        """
        Executes the code and measures fitness using exec() function.
        Any error in the execution results in a infi error
        :param code:
        :param increase_cache_count:
        :return:
        """
        if code in self.fitness_cache:
            if increase_cache_count:
                self.iteration_cache_use_count += 1
            return self.fitness_cache[code]
        # print("code")
        # print("------------------------------------------------------")
        # print(code)
        # print("------------------------------------------------------")

        final_code = self.helper_code.replace("<insertCodeHere>", code)

        error = 0.0

        self.error_programs_total += 1
        signal.alarm(1)
        # signal.setitimer(signal.ITIMER_REAL, 0.1)
        try:
            loc = {}
            exec(final_code, loc, loc)
            error = loc['quality']
        except TimeoutException:
            # print("Timeout exception")
            error = 9999999
            self.error_counts += 1
        except Exception as e:
            # print("Error in code")
            # print(e)
            # print("code")
            # print("------------------------------------------------------")
            # print(code)
            # print("------------------------------------------------------")
            # print("finalcode")
            # print("------------------------------------------------------")
            # print(final_code)
            # print("------------------------------------------------------")
            error = 9999999
            self.error_counts += 1

        # try:
        signal.alarm(0)
        # except Exception as e:
        #     signal.signal(signal.SIGALRM, timeout_handler)

        self.fitness_cache[code] = -error

        return -error

    @staticmethod
    def calc_min(a, b):
        if a < b:
            return a
        return b

    @staticmethod
    def insert_tabs(code):
        """
        Correctly indents the code to the formatting used in _Helper.txt
        :param code:
        :return:
        """
        flag = False
        indentation = 0
        newcode = ""
        while not flag:
            new_line_index = code.find("\n")
            if new_line_index < 0:
                # flag = True
                break
            else:
                # print(code[new_line_index+2])
                if code[new_line_index - 2:new_line_index] == '{:':
                    indentation += 1
                    code = code[:new_line_index - 2] + code[new_line_index:]
                    new_line_index -= 2
                elif code[new_line_index + 1:new_line_index + 3] == ':}':
                    indentation -= 1
                    code = code[:new_line_index + 1] + code[new_line_index + 3:]
                    # new_line_index -= 2
                tabs = ""
                for _ in range(0, indentation+1):
                    tabs += "  "
                newcode += code[:new_line_index] + "\n" + tabs
                code = code[new_line_index + 1:]
        newcode += code
        return newcode

    @staticmethod
    def python_filter(txt):
        """ Create correct python syntax.
        We use {: and :} as special open and close brackets, because
        it's not possible to specify indentation correctly in a BNF
        grammar without this type of scheme."""
        print("nonindented code")
        print(txt)
        indent_level = 0
        tmp = txt[:]
        i = 0
        while i < len(tmp):
            tok = tmp[i:i + 2]
            if tok == "{:":
                indent_level += 1
            elif tok == ":}":
                indent_level -= 1
            tabstr = "\n" + "  " * indent_level
            if tok == "{:" or tok == ":}":
                tmp = tmp.replace(tok, tabstr, 1)
            i += 1
        # Strip superfluous blank lines.
        txt = "\n".join([line for line in tmp.split("\n")
                         if line.strip() != ""])
        return txt


class TerminalType(Enum):
    TERMINAL = 0
    NONTERMINAL = 1


class Production:
    # lhs is a nonterminal
    # rhs is an OrGroup of Sentences
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs


class OrGroup:
    def __init__(self):
        self.expressions = []


class Sentence:
    def __init__(self):
        self.objects = []
        self.object_types = []

    def append_object(self, t_object, object_type):
        self.objects.append(t_object)
        self.object_types.append(object_type)

    def remove_object_at_index(self, index):
        self.objects.remove(index)
        self.object_types.remove(index)

    def insert_object_at_index(self, t_object, object_type, index):
        self.objects.insert(index, t_object)
        self.object_types.insert(index, object_type)

    def __hash__(self):
        t_str = ""
        for obj in self.objects:
            t_str = t_str + obj
        return hash(t_str)

    def __eq__(self, other):
        return (self.__hash__()) == (other.__hash__())

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)


class BracketGroup:
    def __init__(self, expression):
        self.expression = expression
