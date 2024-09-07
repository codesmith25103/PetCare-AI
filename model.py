import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Reading the data
iris = pd.read_csv("/home/lata/Desktop/PetCare-AI/Iris.csv")
print(iris.head())  # Verify the data is loaded correctly

# Correct column names based on your dataset
y = iris['Species']  # Target variable
x = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]  # Features

# Splitting the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Creating and training the model
model = LogisticRegression(max_iter=100)
model.fit(x_train, y_train)

# Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model training complete and saved as 'model.pkl'.")
