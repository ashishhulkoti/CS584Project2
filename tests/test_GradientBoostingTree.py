import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model.GradientBoostingTree import GradientBoostingClassifier, DecisionTreeRegressor

# Fixtures for test data
@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=500, n_classes=2, n_features=5, 
                             n_informative=3, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def multiclass_data():
    X, y = make_classification(n_samples=500, n_classes=3, n_features=6,
                             n_informative=4, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def imbalanced_data():
    X, y = make_classification(n_samples=500, n_classes=2, n_features=5,
                             n_informative=3, weights=[0.9, 0.1], random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Binary classification tests
def test_binary_basic(binary_data):
    """Test basic binary classification functionality"""
    X_train, X_val, y_train, y_val = binary_data
    
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                  max_depth=2, verbose=False)
    gb.fit(X_train, y_train)
    
    # Test predictions shape
    preds = gb.predict(X_val)
    assert preds.shape == (X_val.shape[0],)
    
    # Test probabilities shape
    proba = gb.predict_proba(X_val)
    assert proba.shape == (X_val.shape[0], 2)
    
    # Test accuracy reasonable
    acc = accuracy_score(y_val, preds)
    assert acc > 0.7

def test_binary_early_stopping(binary_data):
    """Test early stopping with binary classification"""
    X_train, X_val, y_train, y_val = binary_data
    
    # Make validation set very small to force early stopping
    X_val, y_val = X_val[:10], y_val[:10]
    
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                  max_depth=2, early_stopping_rounds=5,
                                  verbose=True)  # Set verbose=True to see output
    gb.fit(X_train, y_train, X_val, y_val)
    
    # Early stopping should trigger before reaching n_estimators
    assert gb.best_iteration < 100

def test_binary_feature_subsampling():
    """Test feature subsampling options with binary classification"""
    X, y = make_classification(n_samples=500, n_classes=2, n_features=10,
                             n_informative=8, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test sqrt subsampling
    gb_sqrt = GradientBoostingClassifier(n_estimators=20, max_features='sqrt',
                                       verbose=False)
    gb_sqrt.fit(X_train, y_train)
    preds_sqrt = gb_sqrt.predict(X_val)
    assert preds_sqrt.shape == (X_val.shape[0],)
    
    # Test fixed number subsampling
    gb_fixed = GradientBoostingClassifier(n_estimators=20, max_features=3,
                                        verbose=False)
    gb_fixed.fit(X_train, y_train)
    preds_fixed = gb_fixed.predict(X_val)
    assert preds_fixed.shape == (X_val.shape[0],)
    
    # Test that different subsampling methods produce different results
    assert not np.array_equal(preds_sqrt, preds_fixed)

def test_binary_learning_rate_effect(binary_data):
    """Test that learning rate affects model behavior"""
    X_train, X_val, y_train, y_val = binary_data
    
    # High learning rate
    gb_high = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5,
                                       max_depth=2, verbose=False)
    gb_high.fit(X_train, y_train)
    acc_high = accuracy_score(y_val, gb_high.predict(X_val))
    
    # Low learning rate
    gb_low = GradientBoostingClassifier(n_estimators=50, learning_rate=0.01,
                                     max_depth=2, verbose=False)
    gb_low.fit(X_train, y_train)
    acc_low = accuracy_score(y_val, gb_low.predict(X_val))
    
    # High LR should reach good accuracy faster (with same n_estimators)
    assert acc_high > acc_low or gb_low.best_iteration > gb_high.best_iteration

# Multiclass classification tests
def test_multiclass_basic(multiclass_data):
    """Test basic multiclass classification functionality"""
    X_train, X_val, y_train, y_val = multiclass_data
    
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                  max_depth=2, verbose=False)
    gb.fit(X_train, y_train)
    
    # Test predictions shape
    preds = gb.predict(X_val)
    assert preds.shape == (X_val.shape[0],)
    
    # Test probabilities shape
    proba = gb.predict_proba(X_val)
    assert proba.shape == (X_val.shape[0], 3)
    
    # Test probabilities sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    
    # Test accuracy reasonable
    acc = accuracy_score(y_val, preds)
    assert acc > 0.6

def test_multiclass_early_stopping(multiclass_data):
    """Test early stopping with multiclass classification"""
    X_train, X_val, y_train, y_val = multiclass_data
    
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                  max_depth=2, early_stopping_rounds=5,
                                  verbose=False)
    gb.fit(X_train, y_train, X_val, y_val)
    
    assert gb.best_iteration < 100

# Edge cases and special scenarios
def test_single_class():
    """Test behavior with single class (edge case)"""
    X = np.random.rand(100, 3)  # 100 samples, 3 features
    y_single = np.zeros(100)  # All same class
    
    gb = GradientBoostingClassifier(n_estimators=10, verbose=False)
    gb.fit(X, y_single)
    
    # Verify all predictions are the single class
    preds = gb.predict(X)
    assert np.all(preds == 0)
    
    # Verify probabilities are all 1 for the single class
    if hasattr(gb, 'predict_proba'):
        proba = gb.predict_proba(X)
        assert np.allclose(proba, 1.0)

def test_small_dataset():
    """Test behavior with very small dataset (edge case)"""
    X_small = np.random.rand(20, 2)  # 20 samples, 2 features
    y_small = np.array([0]*10 + [1]*10)  # Binary labels
    
    gb = GradientBoostingClassifier(n_estimators=10, max_depth=1, verbose=False)
    gb.fit(X_small, y_small)
    preds = gb.predict(X_small)
    
    assert preds.shape == (20,)

def test_single_feature():
    """Test behavior with single feature (edge case)"""
    X_single = np.random.rand(100, 1)  # 100 samples, 1 feature
    y_single = (X_single[:, 0] > 0.5).astype(int)  # Simple threshold
    
    gb = GradientBoostingClassifier(n_estimators=10, verbose=False)
    gb.fit(X_single, y_single)
    preds = gb.predict(X_single)
    
    assert preds.shape == (100,)

def test_probability_calibration(imbalanced_data):
    """Test that probabilities are reasonably calibrated"""
    X_train, X_val, y_train, y_val = imbalanced_data
    
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                  max_depth=2, verbose=False)
    gb.fit(X_train, y_train)
    
    proba = gb.predict_proba(X_val)[:, 1]
    mean_proba = proba.mean()
    true_pos_rate = y_val.mean()
    
    # Check mean predicted probability is close to true positive rate
    assert abs(mean_proba - true_pos_rate) < 0.1

def test_predict_proba_consistency(binary_data):
    """Test that predict_proba is consistent with predict"""
    X_train, _, y_train, _ = binary_data
    
    gb = GradientBoostingClassifier(n_estimators=20, verbose=False)
    gb.fit(X_train, y_train)
    
    proba = gb.predict_proba(X_train)
    preds_from_proba = (proba[:, 1] > 0.5).astype(int)
    preds_direct = gb.predict(X_train)
    
    assert np.array_equal(preds_from_proba, preds_direct)