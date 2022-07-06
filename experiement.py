import pickle
import gerryfair
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt

communities_dataset = "./dataset/communities.csv"
communities_attributes = "./dataset/communities_protected.csv"
lawschool_dataset = "./dataset/lawschool.csv"
lawschool_attributes = "./dataset/lawschool_protected.csv"
adult_dataset = "./dataset/adult.csv"
adult_attributes = "./dataset/adult_protected.csv"
student_dataset = "./dataset/student-mat.csv"
student_attributes = "./dataset/student_protected.csv"

C = 10
printflag = True
gamma = .01
fair_model = gerryfair.model.Model(C=C,
                                   printflag=printflag,
                                   gamma=gamma,
                                   fairness_def='FP',
                                   predictor=RandomForestRegressor(max_depth=2))
# predictor=RandomForestRegressor(max_depth=1)
gamma_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
centered = True

# Train Set (Communities)
X, X_prime, y = gerryfair.clean.clean_dataset(communities_dataset, communities_attributes, centered)

warnings.filterwarnings("error")

# Train the model (size=1000, iters=200)
train_size = 500
max_iters = 100
X_train = X.iloc[:train_size]
X_prime_train = X_prime.iloc[:train_size]
y_train = y.iloc[:train_size]
fair_model.set_options(max_iters=max_iters)

something = fair_model.train(X_train, X_prime_train, y_train)

print(something)