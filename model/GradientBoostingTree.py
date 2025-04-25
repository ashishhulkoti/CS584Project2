import numpy as np
import random

class DecisionTreeRegressor:
    """A decision tree regressor for gradient boosting, used as weak learner."""
    
    def __init__(self, max_depth=3, min_samples_split=2, feature_indices=None):
        """
        Initialize the decision tree regressor.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split a node
            feature_indices: Indices of features to consider for splits (None = all features)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_indices = feature_indices  # indices of features to consider for splits
        self.left = None  # Left child node
        self.right = None  # Right child node
        self.feature_index = None  # Index of feature used for splitting
        self.threshold = None  # Threshold value for splitting
        self.value = None  # Prediction value for leaf nodes

    def fit(self, X, y, depth=0):
        """
        Build the decision tree from the training set (X, y).
        
        Args:
            X: Training input samples
            y: Training target values
            depth: Current depth of the tree (used for recursion)
        """
        # Stopping conditions for recursion:
        # 1. Reached max depth
        # 2. Not enough samples to split
        # 3. All target values are the same
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            self.value = np.mean(y, axis=0) if y.ndim > 1 else np.mean(y)
            return

        m, n = X.shape  # Number of samples and features
        # Determine which features to consider for splits
        features = self.feature_indices if self.feature_indices is not None else range(n)
        
        # Initialize variables to track best split
        best_feature, best_threshold, best_error = None, None, float('inf')
        
        # Search for the best split across all features and possible thresholds
        for feature in features:
            thresholds = np.unique(X[:, feature])  # Consider all unique values as potential thresholds
            for threshold in thresholds:
                # Split data based on current feature and threshold
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                # Skip if split results in empty child nodes
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate predictions for both child nodes
                left_value = np.mean(y[left_mask], axis=0) if y.ndim > 1 else np.mean(y[left_mask])
                right_value = np.mean(y[right_mask], axis=0) if y.ndim > 1 else np.mean(y[right_mask])
                
                # Calculate predictions for all samples
                preds = np.where(left_mask[:, None] if y.ndim > 1 else left_mask, left_value, right_value)
                
                # Calculate MSE for this split
                error = np.mean((y - preds) ** 2)
                
                # Update best split if current one is better
                if error < best_error:
                    best_error = error
                    best_feature = feature
                    best_threshold = threshold

        # If no good split found, make this a leaf node
        if best_feature is None:
            self.value = np.mean(y, axis=0) if y.ndim > 1 else np.mean(y)
            return

        # Store best split information
        self.feature_index = best_feature
        self.threshold = best_threshold
        
        # Split the data for child nodes
        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        self.left = DecisionTreeRegressor(self.max_depth, self.min_samples_split, self.feature_indices)
        self.left.fit(X[left_mask], y[left_mask], depth + 1)
        
        self.right = DecisionTreeRegressor(self.max_depth, self.min_samples_split, self.feature_indices)
        self.right.fit(X[right_mask], y[right_mask], depth + 1)

    def predict(self, X):
        """
        Predict target values for input samples X.
        
        Args:
            X: Input samples to predict
            
        Returns:
            Predicted target values
        """
        # If leaf node, return the stored value
        if self.value is not None:
            if isinstance(self.value, np.ndarray):
                return np.tile(self.value, (X.shape[0], 1))  # For multiclass
            else:
                return np.full(X.shape[0], self.value)  # For single output
        
        # Otherwise, split data and predict from child nodes
        mask = X[:, self.feature_index] <= self.threshold
        
        if hasattr(self.left, "predict"):
            left_pred = self.left.predict(X[mask])
            right_pred = self.right.predict(X[~mask])
            
            # Combine predictions from both child nodes
            y_pred = np.empty((X.shape[0], left_pred.shape[1]) if left_pred.ndim > 1 else X.shape[0])
            y_pred[mask] = left_pred
            y_pred[~mask] = right_pred
            return y_pred
        else:
            return np.full(X.shape[0], self.value)


class GradientBoostingClassifier:
    """Gradient Boosting for classification, supporting binary and multiclass problems."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, max_features=None, early_stopping_rounds=None, 
                 tol=1e-4, verbose=False):
        """
        Initialize the gradient boosting classifier.
        
        Args:
            n_estimators: Number of boosting stages (trees) to build
            learning_rate: Shrinks the contribution of each tree
            max_depth: Maximum depth of the individual trees
            min_samples_split: Minimum number of samples required to split a node
            max_features: Number/ratio of features to consider for each split
            early_stopping_rounds: Stop if validation score doesn't improve for this many rounds
            tol: Tolerance for early stopping
            verbose: Controls verbosity of output during training
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.early_stopping_rounds = early_stopping_rounds
        self.tol = tol
        self.verbose = verbose
        self.trees = []  # Will store all the trees
        self.init_pred = None  # Initial prediction (log-odds)
        self.classes_ = None  # Class labels
        self.best_iteration = None  # Best iteration for early stopping

    def _sigmoid(self, x):
        """Sigmoid function for binary classification probabilities."""
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """Softmax function for multiclass probabilities."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerically stable version
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the gradient boosting model.
        
        Args:
            X: Training input samples
            y: Training target values
            X_val: Validation input samples (optional, for early stopping)
            y_val: Validation target values (optional, for early stopping)
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Handle single-class case
        if n_classes == 1:
            self.init_pred = 0.0  # Default prediction for single class
            self.trees = []
            return


        # Initialize model predictions (log-odds)
        if n_classes == 2:
            # Binary classification case
            y_bin = np.where(y == self.classes_[1], 1, 0)  # Convert to binary labels (0/1)
            pos = np.sum(y_bin) + 1e-10  # Positive class count (with smoothing)
            neg = n_samples - pos + 1e-10  # Negative class count (with smoothing)
            self.init_pred = np.log(pos / neg)  # Initial log-odds
            Fm = np.full(n_samples, self.init_pred)  # Current predictions
        else:
            # Multiclass classification case
            y_bin = np.zeros((n_samples, n_classes))  # One-hot encoded labels
            for idx, cls in enumerate(self.classes_):
                y_bin[:, idx] = (y == cls).astype(float)
            
            pos = np.sum(y_bin, axis=0) + 1e-10  # Class counts with smoothing
            neg = n_samples - pos + 1e-10
            self.init_pred = np.log(pos / neg)  # Initial log-odds for each class
            Fm = np.tile(self.init_pred, (n_samples, 1))  # Current predictions

        self.trees = []  # Reset trees
        best_val_loss = float('inf')
        rounds_no_improve = 0
        best_iteration = 0

        # Gradient boosting iterations
        for m in range(self.n_estimators):
            # Feature subsampling (optional)
            if self.max_features is None:
                feature_indices = None  # Use all features
            else:
                # Determine number of features to use
                if self.max_features == 'sqrt':
                    k = max(1, int(np.sqrt(n_features)))
                elif self.max_features == 'log2':
                    k = max(1, int(np.log2(n_features)))
                elif isinstance(self.max_features, int):
                    k = min(n_features, self.max_features)
                else:
                    k = n_features
                feature_indices = sorted(random.sample(range(n_features), k))  # Random feature subset

            if n_classes == 2:
                # Binary classification update
                proba = self._sigmoid(Fm)  # Current probabilities
                residual = y_bin - proba  # Pseudo-residuals (negative gradient)
                
                # Fit a tree to the residuals
                tree = DecisionTreeRegressor(self.max_depth, self.min_samples_split, feature_indices)
                tree.fit(X, residual)
                self.trees.append([tree])  # Store the tree
                
                # Update predictions with the new tree's contribution
                Fm += self.learning_rate * tree.predict(X)
            else:
                # Multiclass classification update (one tree per class)
                proba = self._softmax(Fm)  # Current class probabilities
                residual = y_bin - proba  # Pseudo-residuals for each class
                
                trees_m = []  # Trees for this iteration
                for k in range(n_classes):
                    # Fit a tree to each class's residuals
                    tree = DecisionTreeRegressor(self.max_depth, self.min_samples_split, feature_indices)
                    tree.fit(X, residual[:, k])
                    trees_m.append(tree)
                    
                    # Update predictions for this class
                    Fm[:, k] += self.learning_rate * tree.predict(X)
                self.trees.append(trees_m)  # Store all class trees for this iteration

            # Early stopping check (if validation data provided)
            if X_val is not None and y_val is not None:
                if n_classes == 2:
                    # Binary validation loss calculation
                    val_pred = np.full(X_val.shape[0], self.init_pred)  # Start with initial prediction
                    for trees_m in self.trees:
                        val_pred += self.learning_rate * trees_m[0].predict(X_val)  # Add all trees' contributions
                    
                    val_proba = self._sigmoid(val_pred)
                    # Calculate log loss (binary cross-entropy)
                    val_loss = -np.mean(y_val * np.log(val_proba + 1e-10) + (1 - y_val) * np.log(1 - val_proba + 1e-10))
                else:
                    # Multiclass validation loss calculation
                    val_pred = np.tile(self.init_pred, (X_val.shape[0], 1))  # Start with initial predictions
                    for trees_m in self.trees:
                        for k in range(n_classes):
                            val_pred[:, k] += self.learning_rate * trees_m[k].predict(X_val)  # Add all trees' contributions
                    
                    val_proba = self._softmax(val_pred)
                    # One-hot encode validation labels
                    y_val_onehot = np.zeros((X_val.shape[0], n_classes))
                    for idx, cls in enumerate(self.classes_):
                        y_val_onehot[:, idx] = (y_val == cls).astype(float)
                    # Calculate log loss (categorical cross-entropy)
                    val_loss = -np.mean(np.sum(y_val_onehot * np.log(val_proba + 1e-10), axis=1))
                
                if self.verbose:
                    print(f"Iteration {m+1}, Validation loss: {val_loss:.5f}")
                
                # Check for improvement
                if val_loss + self.tol < best_val_loss:
                    best_val_loss = val_loss
                    rounds_no_improve = 0
                    best_iteration = m + 1
                else:
                    rounds_no_improve += 1
                    # Stop if no improvement for early_stopping_rounds
                    if self.early_stopping_rounds and rounds_no_improve >= self.early_stopping_rounds:
                        if self.verbose:
                            print(f"Early stopping at iteration {m+1}")
                        break
        
        # Store the best iteration (for prediction)
        self.best_iteration = best_iteration if self.early_stopping_rounds else self.n_estimators

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples X.
        
        Args:
            X: Input samples
            
        Returns:
            Array of class probabilities
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # Binary classification probabilities
            Fm = np.full(n_samples, self.init_pred)  # Start with initial prediction
            for trees_m in self.trees[:self.best_iteration]:  # Only use trees up to best_iteration
                Fm += self.learning_rate * trees_m[0].predict(X)  # Add each tree's contribution
            
            proba = self._sigmoid(Fm)  # Convert to probabilities
            return np.vstack([1 - proba, proba]).T  # Return probabilities for both classes
        else:
            # Multiclass probabilities
            Fm = np.tile(self.init_pred, (n_samples, 1))  # Start with initial predictions
            for trees_m in self.trees[:self.best_iteration]:  # Only use trees up to best_iteration
                for k in range(n_classes):
                    Fm[:, k] += self.learning_rate * trees_m[k].predict(X)  # Add each tree's contribution
            
            proba = self._softmax(Fm)  # Convert to probabilities
            return proba

    def predict(self, X):
        """
        Predict class labels for input samples X.
        
        Args:
            X: Input samples
            
        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)  # Get class probabilities
        
        # For binary classification
        if proba.ndim == 1 or proba.shape[1] == 2:
            return self.classes_[(proba[:, 1] > 0.5).astype(int)]  # Threshold at 0.5
        # For multiclass classification
        else:
            indices = np.argmax(proba, axis=1)  # Choose class with highest probability
            return self.classes_[indices]
