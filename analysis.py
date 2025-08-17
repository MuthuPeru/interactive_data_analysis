# analysis.py
# Author: Data Scientist
# Email: 24f1000447@ds.study.iitm.ac.in

import marimo

__generated_with = "0.9.16"
app = marimo.App()


# Cell 1: Import libraries and set up data
# This cell initializes the dataset to be analyzed.
@app.cell
def cell1():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    # Create a simple dataset: x and y with linear relationship + noise
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2.5 * x + np.random.normal(0, 2, 100)
    data = pd.DataFrame({"x": x, "y": y})
    data.head()
    return data, np, pd, plt, x, y


# Cell 2: Create interactive slider widget
# This slider controls the degree of polynomial fit for regression.
@app.cell
def cell2(mo):
    degree_slider = mo.ui.slider(1, 5, value=1, label="Polynomial Degree")
    degree_slider
    return degree_slider,


# Cell 3: Perform regression based on slider
# This depends on data from Cell 1 and the widget from Cell 2.
@app.cell
def cell3(data, np, degree_slider):
    from numpy.polynomial import Polynomial
    deg = degree_slider.value
    # Fit polynomial regression of chosen degree
    coefs = np.polyfit(data["x"], data["y"], deg)
    poly = np.poly1d(coefs)
    y_pred = poly(data["x"])
    (deg, coefs, y_pred[:5])
    return Polynomial, coefs, deg, poly, y_pred


# Cell 4: Plot regression line
# This depends on regression results from Cell 3.
@app.cell
def cell4(plt, data, poly, deg):
    fig, ax = plt.subplots()
    ax.scatter(data["x"], data["y"], label="Data", alpha=0.6)
    ax.plot(data["x"], poly(data["x"]), color="red", label=f"Degree {deg} Fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig
    return ax, fig


# Cell 5: Dynamic markdown explanation
# This markdown output updates dynamically with slider state.
@app.cell
def cell5(mo, deg):
    mo.md(f"""
    ### Analysis Report

    The polynomial regression was fit using **degree = {deg}**.

    - When degree = 1 → simple linear regression.
    - Higher degrees allow capturing more complex patterns, 
      but risk **overfitting**.
    """)
    return


# Entrypoint
if __name__ == "__main__":
    app.run()
