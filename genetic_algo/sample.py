
import click
from ga import GeneticAlgo as ga

import numpy as np

# Pool of genes
pool_scaler = ["MinMax (-1,1)", "MinMax (0,1)", "Std"]
pool_batch_size = [64, 72, 128, 256]
pool_lstm_layers = [[128, 128, 128],
                    [64, 64, 64],
                    [32, 32, 32],
                    [128, 64, 64],
                    [64, 32, 32],
                    [128, 64, 32]]
pool_dropout = [0., 0.1, 0.2]
pool_activation = ['tanh', 'sigmoid']
pool_loss = ['mean_squared_error', 'mae']
# pool_optimizer = ['adam',
#                   optimizers.RMSprop(lr=0.001),
#                   optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
#                   ]
pool_epochs = [5]

genes_dict = dict( scaler = pool_scaler,
                batch_size = pool_batch_size,
                lstm_layers = pool_lstm_layers,
                dropout = pool_dropout,
                activation = pool_activation,
                loss = pool_loss,
                # optimizer = pool_optimizer,
                epochs = pool_epochs
            )

ga = ga(genes_dict, debug=True)

@click.group()
def cli():
    pass

@cli.command(name='make-children')
@click.argument('code1', type=str)
@click.argument('code2', type=str)
@click.option('-n', default=10)
def make_children(code1, code2, n):
    codes = ga.make_children(code1, code2, n)
    print(codes)
    return

@cli.command(name='chromosomes-head')
@click.option('-n', default=5)
def chromosomes_head(n):
    print(ga.chromosomes.head(n))
    # print(ga.chromosomes['lstm_layers'].iloc[0][0])

@cli.command(name='get-population')
@click.option('-n', default=10)
def get_population(n):
    pop = ga.get_initial_population(n)
    print(pop)

@cli.command(name='get-chromosome')
@click.argument('code', type=str)
def get_chromosome(code):
    print(ga.chromosomes.loc[code])

@cli.command(name='survival')
@click.option('-n', default=10)
def survival(n):
    # n=1
    codes = ga.get_initial_population(n)
    np.random.seed(42)
    scores = np.random.randint(0,40,n)
    fitness = dict(zip(codes, scores))
    print(fitness)
    best_parents = ga.get_best_parents(fitness)
    print(best_parents)

@cli.command(name='random')
@click.option('-n', default=5)
def randomizer(n):
    np.random.seed(42)
    a = [1,2,3,4,5,6,7,8,9,10]

    for i in range(n):
        print(np.random.choice(a))

@cli.command(name='test1')
def test1():
    import pandas as pd
    df = pd.DataFrame({'a':[[0,1,3],[2,3,4]], 'b':['a','b'], 'c':[64, 128]})
    # print(df)
    compare = [[0,1,3], 'a', 64]
    for key, row in df.iterrows():
        # print(type(row.values))
        # print(row.values.shape)
        # print(row.values)
        # print(compare)
        # print(row.values==compare)
        # if row.isin(compare).all():        
        if (row.values==compare).all():
            print(key)

    

if __name__ == '__main__':
    cli()
