import psutil
import xgboost as xgb
from datetime import datetime
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import classification_report, accuracy_score

def xgboost(X_train, X_test, y_train, y_test, label_encoder):
    process = psutil.Process()
    dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    params = {"objective": "multi:softmax", "tree_method": "hist", "num_class": 7}

    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

    training_start_time = datetime.now()
    training_start_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    training_start_cpu = process.cpu_percent(interval=None)

    model = xgb.train(params=params, dtrain=dtrain_reg, num_boost_round=500, evals=evals, verbose_eval=0, early_stopping_rounds=50)
    
    training_end_time = datetime.now()
    training_end_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    training_end_cpu = process.cpu_percent(interval=None)

    test_start_time = datetime.now()
    test_start_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    test_start_cpu = process.cpu_percent(interval=None)

    y_pred = model.predict(dtest_reg)

    test_end_time = datetime.now()
    test_end_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    test_end_cpu = process.cpu_percent(interval=None)

    accuracy = accuracy_score(y_test, y_pred)
    gmean = geometric_mean_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder, zero_division=0)

    training_peak_memory = max(training_start_memory, training_end_memory)
    test_peak_memory = max(test_start_memory, test_end_memory)

    training_time = training_end_time - training_start_time
    test_time = test_end_time - test_start_time

    training_memory_usage = training_end_memory - training_start_memory
    test_memory_usage = test_end_memory - test_start_memory

    training_cpu_usage = training_end_cpu - training_start_cpu
    test_cpu_usage = test_end_cpu - test_start_cpu

    print(f"G-Mean - XGBoost: {gmean}")
    print(f"Accuracy - XGBoost: {accuracy}")
    print(f"Training time - XGBoost: {training_time} seconds")
    print(f"Testing time - XGBoost: {test_time} seconds")
    print(f"Training memory usage - XGBoost: {training_memory_usage:.2f} MB")
    print(f"Testing memory usage - XGBoost: {test_memory_usage:.2f} MB")
    print(f"Training CPU usage  - XGBoost: {training_cpu_usage}%")
    print(f"Testing CPU usage  - XGBoost: {test_cpu_usage}%")
    print(f"Training peak memory usage - XGBoost: {training_peak_memory:.2f} MB")
    print(f"Testing peak memory usage - XGBoost: {test_peak_memory:.2f} MB")
    print("Classification Report - XGBoost:")
    print(report)