## Team Members
* Ashish Hulkoti - A20548738
* Harish Hebsur - A20552584

# Project 2

## Boosting Trees

Implement again from first principles the gradient-boosting tree classification algorithm (with the usual fit-predict interface as in Project 1) as described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). Answer the questions below as you did for Project 1. In this assignment, you'll be responsible for developing your own test data to ensure that your implementation is satisfactory. (Hint: Use the same directory structure as in Project 1.)

The same "from first principals" rules apply; please don't use SKLearn or any other implementation. Please provide examples in your README that will allow the TAs to run your model code and whatever tests you include. As usual, extra credit may be given for an "above and beyond" effort.

As before, please clone this repo, work on your solution as a fork, and then open a pull request to submit your assignment. *A pull request is required to submit and your project will not be graded without a PR.*

Put your README below. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?


# Gradient Boosting Classifier

A from-scratch implementation of Gradient Boosting for classification tasks, supporting both binary and multiclass classification. This implementation follows the algorithm described in Sections 10.9-10.10 of "The Elements of Statistical Learning" (2nd Edition).

## Features

- Binary and multiclass classification support
- Early stopping based on validation performance
- Feature subsampling (random forests-style)
- Customizable tree depth and learning rate
- Verbose training output

## Installation and Execution

```bash
# Clone repository
git clone <Repo URL>
cd CS584Project2

# Create a Virtual environment
python -m venv venv

# Switch to virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# To run the Tests
cd GradientBoostingTree/tests
pytest -v
```

## Usage

### Basic Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Binary classification example
print("Binary classification example:")
X, y = make_classification(n_samples=500, n_classes=2, n_features=5, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, 
                              max_features='sqrt', early_stopping_rounds=10, verbose=True)
gb.fit(X_train, y_train, X_val, y_val)

# Evaluate performance
print("Train accuracy:", np.mean(gb.predict(X_train) == y_train))
print("Val accuracy:", np.mean(gb.predict(X_val) == y_val))

# Multiclass classification example
print("\nMulticlass classification example:")
X, y = make_classification(n_samples=500, n_classes=3, n_features=5, 
                          n_informative=3, n_redundant=1, 
                          n_clusters_per_class=1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, 
                              max_features='sqrt', early_stopping_rounds=10, verbose=True)
gb.fit(X_train, y_train, X_val, y_val)

# Evaluate performance
print("Train accuracy:", np.mean(gb.predict(X_train) == y_train))
print("Val accuracy:", np.mean(gb.predict(X_val) == y_val))
```

## Model Overview

### What does the model do?
- Implements gradient boosting with decision trees as weak learners  
- Minimizes logistic loss for binary classification  
- Uses softmax with cross-entropy for multiclass  
- Sequentially builds ensemble that corrects previous errors  

### When should it be used?
- For structured/tabular data with <1000 features  
- When interpretability is important (vs neural networks)  
- When probability estimates are needed  
- When you want to avoid sklearn dependencies  

## Testing Methodology

### How was it tested?

| Test Type | Description | Verification Method |
|-----------|-------------|---------------------|
| Unit Tests | Perfect separation, probability bounds | Assertions |
| Synthetic Data | Binary/multiclass with known patterns | Accuracy checks |
| Numerical Stability | Extreme values, edge cases | Runtime warnings |
| Convergence | Training loss progression | Manual inspection |
| Early Stopping | Validation set performance | Iteration tracking |

**Key Test Cases:**
- Single-class input handling
- High-dimensional data (p > n)
- Constant features
- Linearly separable data
- Imbalanced classes

## Tunable Parameters

| Parameter | Type | Description | Recommended Range |
|-----------|------|-------------|-------------------|
| `n_estimators` | int | Number of trees | 50-500 |
| `learning_rate` | float | Shrinkage factor | 0.01-0.3 |
| `max_depth` | int | Tree depth | 2-6 |
| `min_samples_split` | int | Minimum split size | 2-20 |
| `max_features` | str/int | Feature subsampling | 'sqrt', 'log2', or 0.5-0.8 |
| `early_stopping_rounds` | int | Early stopping patience | 10-50 |
| `tol` | float | Loss tolerance | 1e-4 |
| `verbose` | bool | Print progress | True/False |

## Limitations and Challenges

### Current Limitations

**Input Sensitivity:**
- No native categorical feature handling  
- Requires dense numerical input  

**Performance:**
- No GPU support  
- Memory intensive for deep trees  

**Missing Values:**
- Requires pre-processing  
- No built-in imputation  

### Potential Improvements

| Issue | Possible Solution | Complexity |
|-------|-------------------|------------|
| High dimensions | Feature importance pruning | Medium |
| Categorical data | Optimal binning | High |
| Missing values | Surrogate splits | High |
| Speed | Cython optimization | Medium |

### Fundamental Constraints
- Decision tree expressiveness limits overall model capacity  
- Sequential nature prevents full parallelization  
- Global learning rate affects all trees equally  
