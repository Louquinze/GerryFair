import pickle
import gerryfair
from gerryfair.util import creat_pareto, creat_tracery
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

communities_dataset = "./dataset/communities.csv"  # 0
communities_attributes = "./dataset/communities_protected.csv"
lawschool_dataset = "./dataset/lawschool.csv"  # 1
lawschool_attributes = "./dataset/lawschool_protected.csv"
adult_dataset = "./dataset/adult.csv"  # 3
adult_attributes = "./dataset/adult_protected.csv"
student_dataset = "./dataset/student-mat.csv"  # 5
student_attributes = "./dataset/student_protected.csv"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=int, default=0)
parser.add_argument('--regressor', type=str, default="linear")  # random-forest, gradient-boost, mlp
parser.add_argument('--gamma', type=float, default=.01)
parser.add_argument('--max_iters', type=int, default=200)

if __name__ == '__main__':
    args = parser.parse_args()
    C = 10
    printflag = True
    gamma = args.gamma
    tag = f"{args.regressor}_{args.dataset}_{args.gamma}_{args.max_iters}"
    if args.regressor == "linear":
        predictor = LinearRegression()
    elif args.regressor == "random-forest":
        predictor = RandomForestRegressor(max_depth=3)
    elif args.regressor == "gradient-boost":
        predictor = GradientBoostingRegressor()
    elif args.regressor == "mlp":
        predictor = MLPRegressor()
    fair_model = gerryfair.model.Model(C=C,
                                       printflag=printflag,
                                       gamma=gamma,
                                       fairness_def='FP',
                                       predictor=predictor,
                                       tag=tag)

    centered = True

    # Train Set (Communities)
    if args.dataset == 0:
        X, X_prime, y = gerryfair.clean.clean_dataset(communities_dataset, communities_attributes, centered)
    elif args.dataset == 1:
        X, X_prime, y = gerryfair.clean.clean_dataset(lawschool_dataset, lawschool_attributes, centered)

    warnings.filterwarnings("error")

    # Train the model (size=1000, iters=200)
    train_size = int(len(X)*0.8)
    max_iters = args.max_iters
    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]
    fair_model.set_options(max_iters=max_iters)

    fair_model.train(X_train, X_prime_train, y_train)

    creat_pareto(tag=tag)
    creat_tracery(tag=tag)
