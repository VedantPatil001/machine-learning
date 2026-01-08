import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("placement (3).csv")
x = df.iloc[:, 0:1]
y = df.iloc[:, 1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

from sklearn.metrics import r2_score
y_pred = lr.predict(x_test)
print("Model R² score:", r2_score(y_test, y_pred))

print("Package predictor using CGPA")
cgpa = float(input("Enter your CGPA: "))

if(cgpa<0 or cgpa>10):
    print("Invalid CGPA")
    exit(0)
else:
  new_data = pd.DataFrame({'cgpa': [cgpa]})
  predicted_package = lr.predict(new_data)

  print("Your predicted package is:", predicted_package[0])
  plt.scatter(df['cgpa'],df['package'], label = 'Actual Data')
  plt.plot(x_train, lr.predict(x_train),color = "red", label = "Regression Line")
  plt.scatter(cgpa, predicted_package[0],color = 'green',s=100,label='Your Prediction')
  plt.xlabel('CGPA')
  plt.ylabel('Package (LPA)')
  plt.title('Placement Package Prediction')
  plt.show()
