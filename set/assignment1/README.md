# Assignment 1: Model Training and Testing

This assignment contains three Jupyter notebooks for training, evaluating, and testing machine learning models.

## Notebooks

### `models.ipynb`
Trains two classification models and exports them to ONNX format:
- **BAD Model**: RandomForest classifier trained on biased data (gender labels intentionally flipped)
- **GOOD Model**: AdaBoost classifier with DecisionTree base estimator

Both models are trained on the investigation dataset and saved as ONNX files in the `model/` directory.

### `partitioning_tests.ipynb`
Tests all ONNX models in the `model/` directory for bias across sensitive features:
- Gender (`persoon_geslacht_vrouw`)
- Social benefits (`pla_hist_pla_categorie_doelstelling_16`)
- Medical history features

The notebook evaluates model accuracy across different partitions of these sensitive features and flags significant bias when accuracy differences exceed a threshold (4%).

### `metamorphic_tests.ipynb`
Performs metamorphic testing to assess model robustness:
- **Gender flip test**: Flips gender values (0↔1) and measures prediction changes
- **Age perturbation test**: Increments age by 1 and measures prediction changes

These tests help identify whether models are overly sensitive to small changes in protected attributes.

## Setup

1. **Install dependencies** using Pipenv:
   ```bash
   pipenv install
   ```

2. **Activate the virtual environment**:
   ```bash
   pipenv shell
   ```

3. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

## Running the Notebooks

**Important**: Run the notebooks in this order:

1. **First, run `models.ipynb`** to train and save the models
   - This creates the ONNX model files in the `model/` directory
   - Ensure `data/investigation_train_large_checked.csv` exists

2. **Then run `partitioning_tests.ipynb`** to evaluate bias
   - Tests all models in the `model/` directory
   - Requires `data/investigation_train_large_checked.csv`

3. **Finally, run `metamorphic_tests.ipynb`** for robustness testing
   - Requires `data/synth_data_for_training.csv`
   - Update `MODEL_PATH` in the notebook to test different models

## Data Files

- `data/investigation_train_large_checked.csv`: Main training dataset
- `data/synth_data_for_training.csv`: Synthetic data for metamorphic testing

## Dependencies

All dependencies are specified in `Pipfile`. Key packages include:
- scikit-learn, pandas, numpy
- onnxruntime, skl2onnx, onnx
- jupyter, notebook, ipykernel

