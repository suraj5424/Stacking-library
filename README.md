# StackingEnsemble Class Detailed Documentation

## Class: `StackingEnsemble`
The `StackingEnsemble` class is designed to build multi-layered stacking and blending models, providing a robust framework for ensemble learning, particularly suited for regression tasks. This class allows users to implement two distinct ensemble strategies: **stacking** (with K-fold out-of-fold predictions) and **blending** (with a hold-out validation set). It also includes extensive input validation and error handling to guide the user in case of incorrect inputs or issues during fitting and predicting.

### Parameters:
- **`layers`** : `list of lists`
    - A list of lists, where each inner list contains models (i.e., estimators) for a particular layer in the ensemble.
    - Each model should be a scikit-learn compatible model, meaning it must implement the **`fit()`** and **`predict()`** methods.
    - Example: `[[Model1, Model2], [Model3, Model4]]` would define two layers, with two models in each layer.
    - **Note**: The order of layers matters. Models in later layers will use predictions from models in earlier layers as input features.
    
- **`meta_model`** : `estimator`
    - A single scikit-learn compatible model that combines the predictions from the final layer into a final prediction.
    - This model typically performs regression or classification on the predictions from the previous layer's models (depending on the task).
    - Example: A `LinearRegression()` or `RandomForestRegressor()` might serve as a good meta-model for regression tasks.

- **`n_folds`** : `int`, default=5
    - Specifies the number of folds for **K-fold cross-validation**, which is used for generating out-of-fold predictions during the stacking process.
    - The default is 5, but users can choose any value greater than or equal to **2**.
    - **Note**: Only used when **`blending=False`**.

- **`blending`** : `bool`, default=False
    - If **`True`**, the model uses a hold-out validation set for blending instead of K-fold cross-validation.
    - In blending mode, a portion of the training data is reserved as a hold-out set (specified by **`blend_size`**) and used for training the base models, while predictions for the final meta-model are made on this hold-out set.
    - **Default**: False (indicating stacking mode).

- **`blend_size`** : `float`, default=0.2
    - Specifies the proportion of the training data to hold out for blending (i.e., used as a validation set in blending mode).
    - The value must be between **0 and 1**, where a value of **0.2** means **20%** of the data is used as the hold-out set.
    - **Required**: Only used if **`blending=True`**.

- **`random_state`** : `int`, default=None
    - A seed value for controlling the randomness in splitting the dataset (for cross-validation in stacking or train/hold-out split in blending).
    - **Default**: None (which means the random state is not fixed).

### Attributes:
- **`layer_models_`** : `list`
    - A list that stores the fitted models for each layer after the **`fit()`** method is called.
    - This includes the base models from each layer and their predictions used as inputs for the subsequent layers.

---

### Methods:

#### `__init__(self, layers, meta_model, n_folds=5, blending=False, blend_size=0.2, random_state=None)`
This method initializes the ensemble class and validates input parameters.

##### Parameters:
- **`layers`**, **`meta_model`**, **`n_folds`**, **`blending`**, **`blend_size`**, **`random_state`** (See Parameters section above for detailed descriptions.)

##### Raises:
- **`ValueError`**: If **`layers`** is not a non-empty list of non-empty lists, or if any model in **`layers`** doesn't have **`fit`** or **`predict`** methods.
- **`ValueError`**: If **`meta_model`** doesn't have **`fit`** or **`predict`** methods.
- **`ValueError`**: If **`n_folds`** is less than **2** or not an integer.
- **`ValueError`**: If **`blend_size`** is not between **0 and 1** when **`blending=True`**.

---

#### `fit(self, X, y)`
Fits the stacking ensemble model to the provided training data (**`X`**, **`y`**). This method processes each layer of models and trains them accordingly using either stacking (K-fold CV) or blending (hold-out set).

##### Parameters:
- **`X`**: `pandas.DataFrame` or `numpy.ndarray`
    - The feature matrix containing training data.
- **`y`**: `pandas.Series`, `numpy.ndarray`, or `list`
    - The target vector containing the labels or outputs for each sample.

##### Returns:
- **`self`**: `object`
    - The fitted **`StackingEnsemble`** object.
    
##### Raises:
- **`TypeError`**: If **`X`** is not a pandas DataFrame or numpy array, or if **`y`** is not a pandas Series, numpy array, or list.
- **`ValueError`**: If the number of samples in **`X`** and **`y`** does not match.
- **`RuntimeError`**: If an error occurs during the fitting process, such as failure to split data correctly, errors in model training, or predictions.

##### Workflow:
1. **Input Validation**: Ensures **`X`** and **`y`** are of correct types and dimensions.
2. **Layer-wise Training**:
   - For each layer, the method either generates out-of-fold predictions using K-fold cross-validation (**stacking**) or trains models on the training set and generates predictions on a hold-out set (**blending**).
3. **Final Model Training**: The **meta-model** is trained using the predictions from the final layer as input.

---

#### `predict(self, X)`
Uses the fitted ensemble model to make predictions on new data (**`X`**).

##### Parameters:
- **`X`**: `pandas.DataFrame` or `numpy.ndarray`
    - The feature matrix for which predictions are needed.

##### Returns:
- **`y_pred`**: `numpy.ndarray`
    - The predicted values based on the ensemble model.

##### Raises:
- **`TypeError`**: If **`X`** is not a pandas DataFrame or numpy array.
- **`RuntimeError`**: If an error occurs during prediction (e.g., failure in model predictions).

##### Workflow:
1. **Layer-wise Prediction**: For each layer, predictions are made using the models from that layer.
2. **Meta-Model Prediction**: The final predictions are obtained by passing the predictions from the last layer through the **meta-model**.

---

#### `print_structure(self)`
Prints the entire structure of the ensemble model, including each layer, the models within each layer, and the **meta-model**, in a detailed tree format.

---

## Example Usage

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Define models for the ensemble layers
layer_1_models = [LinearRegression(), RandomForestRegressor(n_estimators=50)]
layer_2_models = [SVR(kernel='rbf', C=1.0, epsilon=0.1)]

# Meta model
meta_model = LinearRegression()

# Create an instance of the StackingEnsemble
ensemble = StackingEnsemble(layers=[layer_1_models, layer_2_models], meta_model=meta_model)

# Example training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the ensemble
ensemble.fit(X_train, y_train)

# Make predictions
y_pred = ensemble.predict(X_test)

# Print the ensemble structure
ensemble.print_structure()
```

##### Example Output:

**Example Output:**

**Stacking Model Structure:**

**Meta Model**: `LinearRegression`
  - **Parameters**: `{'fit_intercept': True, 'normalize': False}`

**Layer 1:**
  - **Model 1**: `LinearRegression`
    - **Parameters**: `{'fit_intercept': True, 'normalize': False}`
  - **Model 2**: `RandomForestRegressor`
    - **Parameters**: `{'n_estimators': 50}`

**Layer 2:**
  - **Model 1**: `SVR`
    - **Parameters**: `{'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}`

**Blending Enabled**: `False`


---

### `_get_changed_params(self, model)`

Returns only the parameters that were explicitly changed by the user for a given model.

- **Parameters**: 
  - `model`: The model whose parameters you want to check.

- **Returns**: 
  - A dictionary of changed parameters or `"No changes (using defaults)"` if no changes were made.

##### Parameters:
- **`model`**: `sklearn` model
    - The model to inspect for changed parameters.

##### Returns:
- **`dict`** or **`str`**: A dictionary of changed parameters (key-value pairs), or a string indicating that no changes were made (i.e., using the default parameters).

##### Example:
```python
{
    'n_estimators': 50
}
