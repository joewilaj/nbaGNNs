This project is an implementation of several graph neural network models for link prediction on the weighted, directed point-differential
graph for the 2013-2019, 2021 seasons for NBA and NCAA basketball. Open src/models.py to select a year and day range for testing or to adjust hyperparameters. Run src/models.py to train and test a model. Predictions for each model for the 2021 season are posted in [predictions](https://github.com/joewilaj/nbaGNNs/tree/main/nbaGNNs/predictions). The prediction printed is (Home Score - Away Score).

The input to the models are graphs representing the state of the season on a given day: The Offense/Defense graph has nba offenses and defenses as nodes, and edges representing interactions between them. The Vegas graph has teams as nodes and its weighted directed edges represent [Vegas point spreads](https://www.kaggle.com/erichqiu/nba-odds-and-scores). The edgeweights in the Offense/Defense graph are computed according to [_Four Factors_](https://www.basketball-reference.com/about/factors.html) statistics from [basketball reference game boxscores](https://www.basketball-reference.com/boxscores/) via [sportsipy](https://github.com/roclark/sportsipy). As a preproccessing step, All graphs are row normalized and The Oracle Adjustment is applied as described in [_An Oracle Method to Predict NFL Games_](http://ramanujan.math.trinity.edu/bmiceli/research/NFLRankings_revised_print.pdf) (2012 Balreira, Miceli, Tegtmeyer) to enhance the random walks on the graphs. 

Next, [_node2vec_](https://arxiv.org/pdf/1607.00653.pdf) (2016, Grover, Leskovec) is applied to the graphs to compute a feature representation of all offense and defense nodes, and all teams in the Vegas graph. Then the graphs along with the node2vec representations are passed to one of 4 graph convolutional layers described in [_Diffusion Convolutional Neural Network_](https://arxiv.org/pdf/1511.02136.pdf) (2016, Atwood, Towsely), [_Design Space for Graph Neural Networks_](https://arxiv.org/pdf/2011.08843.pdf) (2020 Leskovec, Ying, You), [_Graph Neural Networks with Convolutional ARMA Filters_](https://arxiv.org/pdf/1901.01343.pdf) (2021 Bianchi, Grattarola, Livi, Alippi), and [_How Powerful are Graph Neural Networks?_](https://arxiv.org/pdf/1810.00826.pdf) (2019, Hu, Leskovec, Jegelka, Xu). These layers are implemented using [_spektral_](https://github.com/danielegrattarola/spektral).

Now, for a given game, the new representations of both offenses and defenses, along with both teams' representation in the Vegas graph, are passed to a regression neural network to predict the score differential of a game. The model is tested during the selected year and day range, and its win percentage against the spread and against the moneyline are printed along with its MSE for the games in the testing range. 

[Set up the environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using deepnba.yml:

conda env create -f deepnba.yml
