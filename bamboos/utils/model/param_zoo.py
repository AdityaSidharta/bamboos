from hyperopt import hp
from hyperopt.pyll.base import scope
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

regression_param_dict = {
    "Ridge": {
        "alpha": hp.uniform("alpha", 0.01, 10.0),
        "solver": hp.choice(
            "solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        ),
    },
    "Lasso": {"alpha": hp.uniform("alpha", 0.01, 10.0)},
    "ElasticNet": {
        "alpha": hp.uniform("alpha", 0.01, 10.0),
        "l1_ratio": hp.uniform("l1_ratio", 0.01, 0.99),
    },
    "LinearSVR": {
        "epsilon": hp.uniform("epsilon", 0., 0.3),
        "C": hp.uniform("C", 0.1, 5.),
        "loss": hp.choice(
            "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]
        ),
    },
    "KNeighborsRegressor": {
        "n_neighbors": scope.int(hp.quniform("n_neighbours", 1, 100, 1)),
        "algorithm": hp.choice("algorithm", ["ball_tree", "kd_tree", "brute", "auto"]),
        "weights": hp.choice("weights", ["uniform", "distance"]),
        "leaf_size": scope.int(hp.quniform("leaf_size", 5, 100, 1)),
        "p": scope.int(hp.choice("p", [1, 2])),
        "n_jobs": -1,
    },
    "DecisionTreeRegressor": {
        "criterion": hp.choice("criterion", ["mse", "friedman_mse", "mae"]),
        "splitter": hp.choice("splitter", ["best", "random"]),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 1000, 1)),
        "min_samples_split": hp.choice(
            "min_samples_split", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
    },
    "AdaBoostRegressor": {
        "base_estimator": hp.choice(
            "base_estimator", [DecisionTreeRegressor(max_depth=n) for n in range(3, 50)]
        ),
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 1000, 1)),
        "learning_rate": hp.quniform("learning_rate", 0.01, 10., 0.01),
        "loss": hp.choice("loss", ["linear", "square", "exponential"]),
    },
    "BaggingRegressor": {
        "n_jobs": -1,
        "base_estimator": hp.choice(
            "base_estimator",
            [None] + [DecisionTreeRegressor(max_depth=n) for n in range(3, 50)],
        ),
        "n_estimators": 100,
        "max_samples": hp.quniform("max_samples", 0.1, 1.0, 0.01),
        "max_features": hp.quniform("max_features", 0.1, 1.0, 0.01),
        "bootstrap": hp.choice("bootstrap", [True, False]),
        "bootstrap_features": hp.choice("bootstrap_features", [True, False]),
    },
    "ExtraTreesRegressor": {
        "n_jobs": -1,
        "n_estimators": 100,
        "criterion": hp.choice("criterion", ["mae", "mse"]),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 1000, 1)),
        "min_samples_split": hp.choice(
            "min_samples_split", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
        "bootstrap": hp.choice("bootstrap", [True, False]),
    },
    "GradientBoostingRegressor": {
        "loss": hp.choice("loss", ["ls", "lad", "huber", "quantile"]),
        "learning_rate": hp.quniform("learning_rate", 0.01, 0.5, 0.01),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1000, 100)),
        "subsample": hp.quniform("subsample", 0.5, 1.0, 0.1),
        "criterion": hp.choice("criterion", ["friedman_mse", "mse", "mae"]),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 1000, 1)),
        "min_samples_split": hp.choice(
            "min_samples_split", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
        "alpha": hp.quniform("alpha", 0.1, 0.99, 0.01),
    },
    "RandomForestRegressor": {
        "n_jobs": -1,
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1000, 100)),
        "criterion": hp.choice("criterion", ["mae", "mse"]),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 1000, 1)),
        "min_samples_split": hp.choice(
            "min_samples_split", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
        "bootstrap": hp.choice("bootstrap", [True, False]),
    },
    "MLPRegressor": {
        "hidden_layer_sizes": hp.choice(
            "hidden_layer_sizes",
            [
                (n_nodes,) * n_layer
                for n_nodes in range(100, 1001, 50)
                for n_layer in range(1, 4, 1)
            ],
        ),
        "activation": hp.choice("activation", ["logistic", "tanh", "relu"]),
        "solver": hp.choice("solver", ["lbfgs", "sgd", "adam"]),
        "alpha": hp.quniform("alpha", 0.00001, 0.001, 0.00001),
        "learning_rate": hp.choice(
            "learning_rate", ["constant", "invscaling", "adaptive"]
        ),
        "learning_rate_init": hp.quniform("learning_rate_init", 0.001, 1.0, 0.001),
        "max_iter": scope.int(hp.quniform("max_iter", 100, 1000, 100)),
    },
    "XGBoost": {
        "booster": hp.choice("booster", ["gbtree", "gblinear", "dart"]),
        "eta": hp.quniform("eta", 0.01, 1.01, 0.01),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 50, 1)),
        "subsample": hp.quniform("subsample", 0.5, 1.0, 0.1),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1.0, 0.1),
        "max_leaves": scope.int(
            hp.choice("max_leaves", [10 * (2 ** x) for x in range(0, 11, 1)])
        ),
        "num_boost_round": scope.int(hp.quniform("num_boost_round", 100, 1000, 100)),
        "gamma": scope.int(hp.choice("gamma", [0, 1, 2])),
    },
    "LightGBM": {
        "boosting": hp.choice("booster", ["gbdt", "dart"]),
        "eta": hp.quniform("eta", 0.01, 1.01, 0.01),
        "num_leaves": scope.int(
            hp.choice("num_leaves", [10 * (2 ** x) for x in range(0, 11, 1)])
        ),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 50, 1)),
        "min_data_in_leaf": scope.int(hp.quniform("min_data_in_leaf", 1, 100, 1)),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": scope.int(hp.choice("bagging_freq", [0, 1, 2, 3, 4, 5])),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "num_boost_round": scope.int(hp.quniform("num_boost_round", 100, 1000, 100)),
        "min_gain_to_split": scope.int(hp.choice("min_gain_to_split", [0, 1, 2])),
    },
}

binary_param_dict = {
    "LogisticRegression": {
        "penalty": hp.choice("penalty", ["l2"]),
        "C": hp.quniform("C", 0.1, 3.0, 0.1),
        "solver": hp.choice("solver", ["newton-cg", "sag", "lbfgs"]),
        "max_iter": 1000,
    },
    "KNeighborsClassifier": {
        "n_neighbors": scope.int(hp.quniform("n_neighbours", 1, 100, 1)),
        "algorithm": hp.choice("algorithm", ["ball_tree", "kd_tree", "brute", "auto"]),
        "weights": hp.choice("weights", ["uniform", "distance"]),
        "leaf_size": scope.int(hp.quniform("leaf_size", 5, 100, 1)),
        "p": scope.int(hp.choice("p", [1, 2])),
        "n_jobs": -1,
    },
    "DecisionTreeClassifier": {
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "splitter": hp.choice("splitter", ["best", "random"]),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 1000, 1)),
        "min_samples_split": hp.choice(
            "min_samples_split", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
    },
    "AdaBoostClassifier": {
        "base_estimator": hp.choice(
            "base_estimator",
            [DecisionTreeClassifier(max_depth=n) for n in range(1, 50)],
        ),
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 1000, 1)),
        "learning_rate": hp.quniform("learning_rate", 0.01, 10., 0.01),
    },
    "BaggingClassifier": {
        "n_jobs": -1,
        "base_estimator": hp.choice(
            "base_estimator",
            [None] + [DecisionTreeClassifier(max_depth=n) for n in range(1, 50)],
        ),
        "n_estimators": 100,
        "max_samples": hp.quniform("max_samples", 0.1, 1.0, 0.01),
        "max_features": hp.quniform("max_features", 0.1, 1.0, 0.01),
        "bootstrap": hp.choice("bootstrap", [True, False]),
        "bootstrap_features": hp.choice("bootstrap_features", [True, False]),
    },
    "ExtraTreesClassifier": {
        "n_jobs": -1,
        "n_estimators": 100,
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 1000, 1)),
        "min_samples_split": hp.choice(
            "min_samples_split", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
        "bootstrap": hp.choice("bootstrap", [True, False]),
    },
    "GradientBoostingClassifier": {
        "loss": hp.choice("loss", ["deviance", "exponential"]),
        "learning_rate": hp.quniform("learning_rate", 0.01, 0.5, 0.01),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1000, 100)),
        "subsample": hp.quniform("subsample", 0.5, 1.0, 0.1),
        "criterion": hp.choice("criterion", ["friedman_mse", "mse", "mae"]),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 1000, 1)),
        "min_samples_split": hp.choice(
            "min_samples_split", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
    },
    "RandomForestClassifier": {
        "n_jobs": -1,
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1000, 100)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 1000, 1)),
        "min_samples_split": hp.choice(
            "min_samples_split", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
        "bootstrap": hp.choice("bootstrap", [True, False]),
    },
    "MLPClassifier": {
        "hidden_layer_sizes": hp.choice(
            "hidden_layer_sizes",
            [
                (n_nodes,) * n_layer
                for n_nodes in range(100, 1001, 50)
                for n_layer in range(1, 4, 1)
            ],
        ),
        "activation": hp.choice("activation", ["logistic", "tanh", "relu"]),
        "solver": hp.choice("solver", ["lbfgs", "sgd", "adam"]),
        "alpha": hp.quniform("alpha", 0.00001, 0.001, 0.00001),
        "learning_rate": hp.choice(
            "learning_rate", ["constant", "invscaling", "adaptive"]
        ),
        "learning_rate_init": hp.quniform("learning_rate_init", 0.001, 1.0, 0.001),
        "max_iter": scope.int(hp.quniform("max_iter", 100, 1000, 100)),
    },
    "XGBoost": {
        "booster": hp.choice("booster", ["gbtree", "gblinear", "dart"]),
        "eta": hp.quniform("eta", 0.01, 1.01, 0.01),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 50, 1)),
        "subsample": hp.quniform("subsample", 0.5, 1.0, 0.1),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1.0, 0.1),
        "max_leaves": scope.int(
            hp.choice("max_leaves", [10 * (2 ** x) for x in range(0, 11, 1)])
        ),
        "num_boost_round": scope.int(hp.quniform("num_boost_round", 100, 1000, 100)),
        "gamma": scope.int(hp.choice("gamma", [0, 1, 2])),
    },
    "LightGBM": {
        "boosting": hp.choice("booster", ["gbdt", "dart"]),
        "eta": hp.quniform("eta", 0.01, 1.01, 0.01),
        "num_leaves": scope.int(
            hp.choice("num_leaves", [10 * (2 ** x) for x in range(0, 11, 1)])
        ),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 50, 1)),
        "min_data_in_leaf": scope.int(hp.quniform("min_data_in_leaf", 1, 100, 1)),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": scope.int(hp.choice("bagging_freq", [1, 2, 3, 4, 5])),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "num_boost_round": scope.int(hp.quniform("num_boost_round", 100, 1000, 100)),
        "min_gain_to_split": scope.int(hp.choice("min_gain_to_split", [0, 1, 2])),
    },
}


multiclass_param_dict = {
    "LogisticRegression": {
        "penalty": hp.choice("penalty", ["l2"]),
        "C": hp.quniform("C", 0.1, 3.0, 0.1),
        "solver": hp.choice("solver", ["newton-cg", "sag", "lbfgs"]),
        "max_iter": 1000,
    },
    "KNeighborsClassifier": {
        "n_neighbors": scope.int(hp.quniform("n_neighbours", 1, 100, 1)),
        "algorithm": hp.choice("algorithm", ["ball_tree", "kd_tree", "brute", "auto"]),
        "weights": hp.choice("weights", ["uniform", "distance"]),
        "leaf_size": scope.int(hp.quniform("leaf_size", 5, 100, 1)),
        "p": scope.int(hp.choice("p", [1, 2])),
        "n_jobs": -1,
    },
    "DecisionTreeClassifier": {
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "splitter": hp.choice("splitter", ["best", "random"]),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 1000, 1)),
        "min_samples_split": hp.choice(
            "min_samples_split", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
    },
    "ExtraTreesClassifier": {
        "n_jobs": -1,
        "n_estimators": 100,
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 1000, 1)),
        "min_samples_split": hp.choice(
            "min_samples_split", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
        "bootstrap": hp.choice("bootstrap", [True, False]),
    },
    "RandomForestClassifier": {
        "n_jobs": -1,
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1000, 100)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 1000, 1)),
        "min_samples_split": hp.choice(
            "min_samples_split", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
        "bootstrap": hp.choice("bootstrap", [True, False]),
    },
    "MLPClassifier": {
        "hidden_layer_sizes": hp.choice(
            "hidden_layer_sizes",
            [
                (n_nodes,) * n_layer
                for n_nodes in range(100, 1001, 50)
                for n_layer in range(1, 4, 1)
            ],
        ),
        "activation": hp.choice("activation", ["logistic", "tanh", "relu"]),
        "solver": hp.choice("solver", ["lbfgs", "sgd", "adam"]),
        "alpha": hp.quniform("alpha", 0.00001, 0.001, 0.00001),
        "learning_rate": hp.choice(
            "learning_rate", ["constant", "invscaling", "adaptive"]
        ),
        "learning_rate_init": hp.quniform("learning_rate_init", 0.001, 1.0, 0.001),
        "max_iter": scope.int(hp.quniform("max_iter", 100, 1000, 100)),
    },
    "XGBoost": {
        "booster": hp.choice("booster", ["gbtree", "gblinear", "dart"]),
        "eta": hp.quniform("eta", 0.01, 1.01, 0.01),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 50, 1)),
        "subsample": hp.quniform("subsample", 0.5, 1.0, 0.1),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1.0, 0.1),
        "max_leaves": scope.int(
            hp.choice("max_leaves", [10 * (2 ** x) for x in range(0, 11, 1)])
        ),
        "num_boost_round": scope.int(hp.quniform("num_boost_round", 100, 1000, 100)),
        "gamma": scope.int(hp.choice("gamma", [0, 1, 2])),
    },
    "LightGBM": {
        "boosting": hp.choice("booster", ["gbdt", "dart"]),
        "eta": hp.quniform("eta", 0.01, 1.01, 0.01),
        "num_leaves": scope.int(
            hp.choice("num_leaves", [10 * (2 ** x) for x in range(0, 11, 1)])
        ),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 50, 1)),
        "min_data_in_leaf": scope.int(hp.quniform("min_data_in_leaf", 1, 100, 1)),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": scope.int(hp.choice("bagging_freq", [1, 2, 3, 4, 5])),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "num_boost_round": scope.int(hp.quniform("num_boost_round", 100, 1000, 100)),
        "min_gain_to_split": scope.int(hp.choice("min_gain_to_split", [0, 1, 2])),
    },
}
