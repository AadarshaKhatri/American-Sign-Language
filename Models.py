import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def load_serialized_images(file_path='./Images_serialized.pickle'):
    data_folder = pickle.load(open(file_path, 'rb'))
    data = np.asarray(data_folder['data'])
    labels = np.asarray(data_folder['labels'])
    return data, labels

def train_model(data, labels, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model, x_test, y_test

def evaluate_model(model, x_test, y_test):
    predicted_values = model.predict(x_test)
    score = accuracy_score(predicted_values, y_test)
    return score

def save_model(model, file_path='trained_model.p'):
    f = open(file_path, 'wb')
    pickle.dump({'model': model}, f)
    f.close()

def main():
    data, labels = load_serialized_images()
    model, x_test, y_test = train_model(data, labels)
    accuracy = evaluate_model(model, x_test, y_test)
    print('{}% of samples were classified correctly!'.format(accuracy * 100))
    save_model(model)

if __name__ == "__main__":
    main()
