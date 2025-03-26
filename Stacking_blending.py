import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split

class StackingEnsemble:
    """
    A robust ensemble class for building multi-layer stacking and blending models,
    particularly designed for regression tasks.
    
    This class supports both stacking (using K-fold out-of-fold predictions) and
    blending (using a hold-out validation set). It includes input validation and
    error handling to provide informative messages when issues arise.

    Parameters:
    -----------
    layers : list of lists
        Each element should be a list of scikit-learn compatible models for that layer.
        Each model must implement fit() and predict() methods.
    meta_model : estimator
        A scikit-learn compatible model used to combine the outputs from the final layer.
    n_folds : int, default=5
        Number of folds for generating out-of-fold predictions (used in stacking mode).
    blending : bool, default=False
        If True, use blending (hold-out approach) instead of stacking.
    blend_size : float, default=0.2
        Proportion of the training data to hold out for blending (only used if blending=True).
    random_state : int, default=None
        Seed for reproducibility.
    """

    def __init__(self, layers, meta_model, n_folds=5, blending=False, blend_size=0.2, random_state=None):
        # Validate layers: should be a non-empty list of non-empty lists
        if not isinstance(layers, list) or not layers or not all(isinstance(l, list) and l for l in layers):
            raise ValueError("`layers` must be a non-empty list of non-empty lists of models.")
        # Validate that each model in layers implements fit and predict
        for layer in layers:
            for model in layer:
                if not (hasattr(model, "fit") and hasattr(model, "predict")):
                    raise ValueError("Each model in layers must implement fit() and predict().")
        
        # Validate meta_model
        if not (hasattr(meta_model, "fit") and hasattr(meta_model, "predict")):
            raise ValueError("`meta_model` must implement fit() and predict().")
        
        # Validate n_folds and blend_size
        if not isinstance(n_folds, int) or n_folds < 2:
            raise ValueError("`n_folds` must be an integer greater than or equal to 2.")
        if blending:
            if not (0.0 < blend_size < 1.0):
                raise ValueError("`blend_size` must be a float between 0 and 1.")
        
        self.layers = layers
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.blending = blending
        self.blend_size = blend_size
        self.random_state = random_state
        self.layer_models_ = []

    def fit(self, X, y):
        """
        Fit the ensemble using the training data X and target y.

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.array
            Feature matrix.
        y : pandas.Series or numpy.array
            Target vector.
            
        Returns:
        --------
        self : object
            Fitted estimator.
        """
        # Validate X and y types and dimensions
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame or a numpy array.")
        if not isinstance(y, (pd.Series, np.ndarray, list)):
            raise TypeError("y must be a pandas Series, numpy array, or list.")

        try:
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            if not hasattr(y, 'iloc'):
                y = pd.Series(y)
        except Exception as e:
            raise ValueError(f"Error converting inputs to pandas objects: {e}")

        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be the same.")

        # For blending, split the data into training and hold-out sets.
        if self.blending:
            try:
                X_train, X_hold, y_train, y_hold = train_test_split(
                    X, y, test_size=self.blend_size, random_state=self.random_state
                )
            except Exception as e:
                raise RuntimeError(f"Error during train/hold split: {e}")
        else:
            X_current = X.copy()

        # Process each layer
        for layer_idx, layer in enumerate(self.layers):
            n_models = len(layer)
            fitted_models = []
            if self.blending:
                # Prepare arrays to store predictions on training and hold-out sets.
                train_preds = np.zeros((X_train.shape[0], n_models))
                hold_preds  = np.zeros((X_hold.shape[0], n_models))
                for model_idx, model in enumerate(layer):
                    # For hold-out predictions, train on current training set and predict on hold-out.
                    try:
                        cloned_model = clone(model)
                        cloned_model.fit(X_train, y_train)
                        hold_preds[:, model_idx] = cloned_model.predict(X_hold)
                    except Exception as e:
                        raise RuntimeError(f"Error in blending at layer {layer_idx+1}, model {model_idx+1} (hold-out): {e}")
                    
                    # Train the model on the same training set.
                    try:
                        model.fit(X_train, y_train)
                    except Exception as e:
                        raise RuntimeError(f"Error training model at layer {layer_idx+1}, model {model_idx+1}: {e}")
                    fitted_models.append(model)
                    # Get training set predictions.
                    try:
                        train_preds[:, model_idx] = model.predict(X_train)
                    except Exception as e:
                        raise RuntimeError(f"Error predicting on training set at layer {layer_idx+1}, model {model_idx+1}: {e}")
                # Update X_train and X_hold to be the meta-features for the next layer.
                X_train = pd.DataFrame(
                    train_preds,
                    columns=[f"Layer{layer_idx+1}_Model{m+1}" for m in range(n_models)]
                )
                X_hold = pd.DataFrame(
                    hold_preds,
                    columns=[f"Layer{layer_idx+1}_Model{m+1}" for m in range(n_models)]
                )
            else:
                # Stacking mode using K-Fold out-of-fold predictions.
                oof_preds = np.zeros((X_current.shape[0], n_models))
                kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                for model_idx, model in enumerate(layer):
                    oof_model_preds = np.zeros(X_current.shape[0])
                    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_current)):
                        try:
                            X_train_fold = X_current.iloc[train_idx]
                            X_valid_fold = X_current.iloc[valid_idx]
                            y_train_fold = y.iloc[train_idx]
                        except Exception as e:
                            raise RuntimeError(f"Error splitting data in fold {fold+1} of layer {layer_idx+1}: {e}")
                        try:
                            cloned_model = clone(model)
                            cloned_model.fit(X_train_fold, y_train_fold)
                            oof_model_preds[valid_idx] = cloned_model.predict(X_valid_fold)
                        except Exception as e:
                            raise RuntimeError(f"Error in fold {fold+1} at layer {layer_idx+1}, model {model_idx+1}: {e}")
                    oof_preds[:, model_idx] = oof_model_preds
                    try:
                        model.fit(X_current, y)
                    except Exception as e:
                        raise RuntimeError(f"Error training full-data model at layer {layer_idx+1}, model {model_idx+1}: {e}")
                    fitted_models.append(model)
                X_current = pd.DataFrame(
                    oof_preds,
                    columns=[f"Layer{layer_idx+1}_Model{m+1}" for m in range(n_models)]
                )
            self.layer_models_.append(fitted_models)
        
        # Train the meta model using the final layer's predictions.
        try:
            if self.blending:
                self.meta_model.fit(X_hold, y_hold)
            else:
                self.meta_model.fit(X_current, y)
        except Exception as e:
            raise RuntimeError(f"Error training meta model: {e}")
        
        return self

    def predict(self, X):
        """
        Make predictions using the fitted ensemble.

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.array
            Feature matrix.

        Returns:
        --------
        y_pred : numpy.array
            Predicted values.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame or a numpy array.")
        try:
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
        except Exception as e:
            raise ValueError(f"Error converting X to DataFrame: {e}")

        X_current = X.copy()

        for layer_idx, fitted_models in enumerate(self.layer_models_):
            try:
                layer_preds = np.column_stack([model.predict(X_current) for model in fitted_models])
            except Exception as e:
                raise RuntimeError(f"Error during prediction at layer {layer_idx+1}: {e}")
            X_current = pd.DataFrame(
                layer_preds,
                columns=[f"Layer{layer_idx+1}_Model{m+1}" for m in range(len(fitted_models))]
            )
        
        try:
            return self.meta_model.predict(X_current)
        except Exception as e:
            raise RuntimeError(f"Error during meta model prediction: {e}")
    def print_structure(self):
        """
        Prints the entire stacking model structure in a detailed tree format,
        including model names and only explicitly changed parameters.
        """
        print("\nStacking Model Structure:")
        print("└── Meta Model: ", self.meta_model.__class__.__name__)
        print("    │   Parameters:", self._get_changed_params(self.meta_model))
    
        for layer_idx, layer_models in enumerate(self.layers):
            print(f"    ├── Layer {layer_idx + 1}:")
            for model_idx, model in enumerate(layer_models):
                print(f"    │   ├── Model {model_idx + 1}: {model.__class__.__name__}")
                print(f"    │   │   Parameters: {self._get_changed_params(model)}")
        
        print("Blending Enabled: ", self.blending)
    
    def _get_changed_params(self, model):
        """
        Returns only the parameters that were explicitly changed by the user.
        This function manually compares the default parameters with the user-set parameters.
        """
        # Get the default parameters for the model (assuming they are defined in the class)
        model_class = model.__class__
        default_params = model_class().get_params()
    
        # Get the user-defined parameters
        current_params = model.get_params()
    
        # Identify the changed parameters (non-default)
        changed_params = {}
        for param, value in current_params.items():
            if param in default_params and value != default_params[param]:
                changed_params[param] = value
    
        # Return the changed parameters, or indicate if none have been changed
        return changed_params if changed_params else "No changes (using defaults)"
