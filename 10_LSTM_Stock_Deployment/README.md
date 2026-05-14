# Assignment 10: LSTM Stock Forecasting and Flask Deployment

## Files
- `10_lstm_stock_deployment.py`
- `stock_data.csv`
- `lstm_model.h5`, if TensorFlow is installed and the model is trained

## Packages Required
```bash
python -m pip install pandas numpy matplotlib scikit-learn flask
```

Optional package for the original LSTM model:
```bash
python -m pip install tensorflow
```

The script also runs without TensorFlow by using a scikit-learn fallback model.

Or install the common project requirements from the project root:
```bash
python -m pip install -r requirements.txt
```

## How to Run Prediction Demo
From the project root:
```bash
python 10_LSTM_Stock_Deployment/10_lstm_stock_deployment.py
```

## How to Run Flask App
From the project root:
```bash
RUN_FLASK=1 python 10_LSTM_Stock_Deployment/10_lstm_stock_deployment.py
```

If using VS Code, select the interpreter from your local `.venv` folder.

