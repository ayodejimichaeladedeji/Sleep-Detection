import psutil
from datetime import datetime

from sklearn.ensemble import VotingClassifier
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def voting_classifier(X_train, X_test, y_train, y_test, label_encoder):

    process = psutil.Process()
    model1 = GradientBoostingClassifier()
    model2 = RandomForestClassifier()
    ensemble_model = VotingClassifier(estimators=[('gb', model1), ('rf', model2)], voting='hard')

    training_start_time = datetime.now()
    training_start_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    training_start_cpu = process.cpu_percent(interval=None)

    ensemble_model.fit(X_train, y_train)

    training_end_time = datetime.now()
    training_end_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    training_end_cpu = process.cpu_percent(interval=None)

    test_start_time = datetime.now()
    test_start_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    test_start_cpu = process.cpu_percent(interval=None)

    y_pred_ensemble = ensemble_model.predict(X_test)

    test_end_time = datetime.now()
    test_end_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    test_end_cpu = process.cpu_percent(interval=None)

    training_peak_memory = max(training_start_memory, training_end_memory)
    test_peak_memory = max(test_start_memory, test_end_memory)

    training_time = training_end_time - training_start_time
    test_time = test_end_time - test_start_time

    training_memory_usage = training_end_memory - training_start_memory
    test_memory_usage = test_end_memory - test_start_memory

    training_cpu_usage = training_end_cpu - training_start_cpu
    test_cpu_usage = test_end_cpu - test_start_cpu

    accuracy = accuracy_score(y_test, y_pred_ensemble)
    gmean = geometric_mean_score(y_test, y_pred_ensemble)
    report = classification_report(y_test, y_pred_ensemble)

    print(f"G-Mean - Hard voting - RF & GB: {gmean}")
    print(f"Accuracy - Hard voting - RF & GB: {accuracy}")
    print(f"Training time - Hard voting - RF & GB: {training_time} seconds")
    print(f"Testing time - Hard voting - RF & GB: {test_time} seconds")
    print(f"Training memory usage - Hard voting - RF & GB: {training_memory_usage:.2f} MB")
    print(f"Testing memory usage - Hard voting - RF & GB: {test_memory_usage:.2f} MB")
    print(f"Training CPU usage  - Hard voting - RF & GB: {training_cpu_usage}%")
    print(f"Testing CPU usage  - Hard voting - RF & GB: {test_cpu_usage}%")
    print(f"Training peak memory usage - Hard voting - RF & GB: {training_peak_memory:.2f} MB")
    print(f"Testing peak memory usage - Hard voting - RF & GB: {test_peak_memory:.2f} MB")
    print("Classification Report - Hard voting - RF & GB:")
    print(report)