import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


# Create pipeline
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(**kwargs))


x = np.arange(0, 10, 0.1).reshape(-1, 1)
y = np.sin(x) + 0.5 * x + np.random.normal(0, 0.1, x.shape)

# Check our data
plt.style.use('bmh')
plt.scatter(x, y)
plt.show()

# Дooking for the best params
param_grid = {'polynomialfeatures__degree': np.arange(21), 'linearregression__fit_intercept': [True, False]}
grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
grid.fit(x, y)
model = grid.best_estimator_
print(grid.best_params_)

plt.scatter(x, y)
plt.plot(x, model.predict(x), c='red')
plt.show()
