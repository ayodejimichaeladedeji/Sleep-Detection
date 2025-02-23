import psutil
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def random_forest(X_train, X_test, y_train, y_test, label_encoder):
    process = psutil.Process()
    model = RandomForestClassifier(class_weight='balanced', random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=0)

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    print(f"Best parameters - RF: {best_rf}")

    training_start_time = datetime.now()
    training_start_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    training_start_cpu = process.cpu_percent(interval=None)

    best_rf.fit(X_train, y_train)

    training_end_time = datetime.now()
    training_end_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    training_end_cpu = process.cpu_percent(interval=None)

    test_start_time = datetime.now()
    test_start_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    test_start_cpu = process.cpu_percent(interval=None)

    y_pred = best_rf.predict(X_test)

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

    print(f"G-Mean - Random Forest: {gmean}")
    print(f"Accuracy - Random Forest: {accuracy}")
    print(f"Training time - Random Forest: {training_time} seconds")
    print(f"Testing time - Random Forest: {test_time} seconds")
    print(f"Training memory usage - Random Forest: {training_memory_usage:.2f} MB")
    print(f"Testing memory usage - Random Forest: {test_memory_usage:.2f} MB")
    print(f"Training CPU usage  - Random Forest: {training_cpu_usage}%")
    print(f"Testing CPU usage  - Random Forest: {test_cpu_usage}%")
    print(f"Training peak memory usage - Random Forest: {training_peak_memory:.2f} MB")
    print(f"Testing peak memory usage - Random Forest: {test_peak_memory:.2f} MB")
    print("Classification Report - Random Forest:")
    print(report)
