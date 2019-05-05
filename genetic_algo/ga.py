import ast, random, itertools

import numpy as np 
import pandas as pd 

from collections.abc import Iterable

class GeneticAlgo():

    def __init__(self, genes_dict, debug=False, random_seed=42):
        self.debug = debug
        self.random_seed = random_seed
        self.genes_dict = genes_dict
        # self.genes_dict = self.prepare_data(genes_dict)
        self.generate_chromosomes()

    # def prepare_data(self, genes_dict):
    #     '''Handle iterables'''
    #     is_iterable = {}
    #     if self.debug:
    #         print('Processing data...')
    #     for key, val in genes_dict.items():
    #         condition = type(val) != str and isinstance(val, Iterable)
    #         if condition:
    #             genes_dict[key] = [str(i) for i in genes_dict[key]]
    #         is_iterable[key] = condition
    #     if self.debug:
    #         print('Processing done.')
    #     self.is_iterable = is_iterable
    #     return genes_dict

    def generate_chromosomes(self):
        keys = self.genes_dict.keys()
        values = self.genes_dict.values()
        chromosomes = list(itertools.product(*values))
        c_len = len(chromosomes)
        codes = [str(i).zfill(5) for i in range(c_len)]
        # self.gene_categories = keys
        # chromosomes = dict(zip(codes,chromosomes))
        chromosomes = pd.DataFrame(chromosomes, index=codes, columns=keys)
        self.chromosomes = chromosomes
        if self.debug:
            print('Number of Chromosomes: {}'.format(chromosomes.shape[0]))

    def get_initial_population(self, n):
        np.random.seed(self.random_seed)
        codes = list(self.chromosomes.index)
        return np.random.choice(codes, n, replace=False)

    def get_best_parents(self, fitness):
        '''
        fitness - dict {code: score}
        '''
        result = sorted(fitness.items(), key=lambda x: x[1], reverse=True)[:2]
        result = [i[0] for i in result] # get code
        # duplicate single parent
        if len(result) == 1:
            result = result*2
        return result

    def make_children(self, parent1_code, parent2_code, num_children, mutation_prob=0.1):
        # random seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        parent_prob = (1 - mutation_prob)/2
        children_chromosomes = []
        for i in range(num_children):
            chromosome = []
            for key in self.chromosomes:
                choices = [self.chromosomes.loc[parent1_code, key], 
                            self.chromosomes.loc[parent2_code, key],
                            random.choice(self.genes_dict[key])]
                # print(choices)                
                if type(choices[0]) != str and isinstance(choices[0], Iterable):
                    # print(choices[0])
                    # print('{} is Iterable'.format(key))
                    choices = [str(i) for i in choices]
                    gene = [np.random.choice(choices, p = [parent_prob, parent_prob, mutation_prob])][0]
                    gene = ast.literal_eval(gene)
                else:
                    gene = [np.random.choice(choices, p = [parent_prob, parent_prob, mutation_prob])][0]
                
                # gene = [np.random.choice(choices, p = [parent_prob, parent_prob, mutation_prob])][0]

                chromosome.append(gene)
            children_chromosomes.append(chromosome)

        codes = list(map(self.get_chromosome_code, children_chromosomes))
        codes = list(set(codes))
        return codes

    def get_chromosome_code(self, chromosome):
        c = self.chromosomes
        for key, row in c.iterrows():
            # if row.isin(chromosome).all():
            if (row.values.tolist()==chromosome):        
                break
        return key

    
    



