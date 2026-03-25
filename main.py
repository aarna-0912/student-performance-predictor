# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("C:/Users/Window/Desktop/BYOP_Project/student_data.csv")

# Input (X) and Output (y)
X = data[["StudyHours", "Attendance"]]
y = data["Marks"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Take user input
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))

# Predict
prediction = model.predict([[hours, attendance]])

print(f"\n📊 Predicted Marks: {prediction[0]:.2f}")

if prediction[0] >= 80:
    print("Performance: Excellent 🎉")
elif prediction[0] >= 60:
    print("Performance: Good 👍")
else:
    print("Performance: Needs Improvement 📚")

import matplotlib.pyplot as plt

# Plot graph
plt.scatter(data["StudyHours"], data["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")

plt.show()
