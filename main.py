import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pickle

def main():
    df = pd.read_csv('./Q7.csv')
    print(df.columns)
    # Split the data into features (X) and target variable (y)
    X = df.drop(['label', 'filename'], axis=1)
    y = df['label']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Applying Standard Normalization method (Z-Score Equation):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handling missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train_scaled)
    X_test = imputer.transform(X_test_scaled)


    # Create a Random Forest Classifier model
    model = KNeighborsClassifier(n_neighbors=6)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy*100, '%')

    file_path = './music_genre_model.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    main()