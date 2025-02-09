import numpy as np
import wandb
from gplearn.genetic import SymbolicRegressor
from sklearn.utils import check_random_state

# init wandb
wandb.login(key="1a52f6079ddb0d4c0e9f5869d9cc0bdd3f5d9a01")
wandb.init(project='genetic_programming_analysis', name='Koza_3')


data_train = np.load('Datasets/Koza_3_train.npy')
data_test = np.load('Datasets/Koza_3_test.npy')

X_train = data_train[:,0].reshape(-1,1)
y_train = data_train[:,1]

X_test = data_test[:,0].reshape(-1,1)
y_test = data_test[:,1]


rmse_trains = []
rmse_tests = []
lengths = []

for _ in range(10):
    gp = SymbolicRegressor(function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'sin', 'cos', 'tan'))
    gp.fit(X_train, y_train)

    # save the model
    y_pred_train = gp.predict(X_train)
    rmse_train = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
    rmse_trains.append(rmse_train)

    y_pred_test = gp.predict(X_test)
    rmse_test = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    rmse_tests.append(rmse_test)

    # save the length of the program
    lengths.append(gp._program.length_)

    print(f"RMSE Train: {rmse_train}, RMSE Test: {rmse_test}, Length: {gp._program.length_}")


mean_rmse_train = np.mean(rmse_trains)
std_rmse_train = np.std(rmse_trains)

mean_rmse_test = np.mean(rmse_tests)
std_rmse_test = np.std(rmse_tests)

mean_length = np.mean(lengths)
std_length = np.std(lengths)

# record the metrics
wandb.log({
    'Mean RMSE Train': mean_rmse_train,
    'Std RMSE Train': std_rmse_train,
    'Mean RMSE Test': mean_rmse_test,
    'Std RMSE Test': std_rmse_test,
    'Mean Program Length': mean_length,
    'Std Program Length': std_length
})

print(f"Mean RMSE Train: {mean_rmse_train}, Standard Deviation: {std_rmse_train}")
print(f"Mean RMSE Test: {mean_rmse_test}, Standard Deviation: {std_rmse_test}")
print(f"Mean Length: {mean_length}, Standard Deviation: {std_length}")

wandb.finish()


