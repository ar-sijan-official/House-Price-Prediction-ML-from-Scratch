# House Price Prediction: ML from Scratch 🏠

### **Why build it from scratch?**
Most people start their Machine Learning journey by simply calling `.fit()` from a library and moving on. I decided to take the "hard road." This project is a ground-up implementation of **Multivariate Linear Regression** using only **NumPy**. No Scikit-Learn "magic"—just pure math, vectorized matrix operations, and an obsession with how gradients actually move.

---

##  Project Intent
The goal of this project was to demystify the "Black Box" of ML models. By manually coding the optimization logic, I’ve mastered the fundamental bridge between abstract mathematical theory and computational efficiency.

### **Core Learning Pillars:**
* **Vectorized Thinking:** Moving away from slow `for` loops to high-performance Matrix Multiplication ($X \cdot w + b$).
* **Mathematical Optimization:** Implementing **Gradient Descent** from scratch to find the global minimum of the Cost Function.
* **Feature Engineering:** Handling real-world data issues like categorical encoding for `Location` and `Condition`, and implementing **Z-score Normalization** to ensure stable convergence.
* **The Judge (Cost Function):** Using **Mean Squared Error (MSE)** to mathematically quantify and track model performance.

---

##  The Math Under the Hood

To truly understand the learner, I implemented the following:

1. **The Hypothesis:** $$f_{w,b}(x) = wx + b$$

2. **The Cost Function (MSE):**
   $$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

3. **The Gradient Descent Update Rule:**
   $$w = w - \alpha \frac{\partial}{\partial w} J(w,b)$$



---

##  Tech Stack & Methodology
* **NumPy:** For vectorized linear algebra operations and parameter management.
* **Pandas:** For data ingestion, cleaning, and preprocessing.
* **Matplotlib:** To visualize the "Learning Curve" (Cost vs. Iterations).
* **Python:** The core engine for logic implementation.

---

##  How It Works
1. **Data Cleaning:** Automatically drops irrelevant identifiers (like `Id`) and prepares the features.
2. **Scaling:** Normalizes features so that large values (like `Area`) don't dominate smaller counts (like `Bedrooms`) during optimization.
3. **Training:** The Gradient Descent engine runs iteratively, updating weights and biases until the cost stabilizes (convergence).
4. **Inference:** Includes an interactive prediction tool that takes raw house details, applies the learned weights, and outputs a dollar-value prediction.

---

##  Results
The model demonstrates a clear downward trajectory in the cost history plot, proving that the Gradient Descent logic is correctly converging on the optimal parameters.

> **Note:** This project isn't just about predicting prices; it's a testament to understanding the fundamental mechanics of Artificial Intelligence.

---

###  Repository Structure
* `house_data.csv`: The dataset used for training.
* `model.ipynb`: Jupyter Notebook containing the step-by-step implementation.
* `README.md`: Project documentation.

---
*Created as a milestone in my Machine Learning journey.*
