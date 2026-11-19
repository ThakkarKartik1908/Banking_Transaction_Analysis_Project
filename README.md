# Banking Transaction Analysis - Real World Case Study
Python for Data Science (BE05000231) | SVIT, Vasad (GTU)
Student: Kartik Thakkar (240410107057)

## Files
| File | Purpose |
|---|---|
| `Banking_Transaction_Analysis.ipynb` | Notebook with all 21 steps (open in Google Colab / Jupyter) |
| `banking_transaction_analysis.py` | Same code as a plain script (`python banking_transaction_analysis.py`) |
| `generate_dataset.py` | Creates `banking_transactions.csv` (synthetic data, PBL schema + `IsFraud` label) |
| `banking_transactions.csv` | The dataset used in the report |
| `requirements.txt` | Python packages |
| `outputs/` | Figures (`figures/`), `results.json`, cleaned data, customer segments, flagged anomalies, model comparison, run log |

## Run
```
pip install -r requirements.txt
python generate_dataset.py                 # optional - the CSV is already included
python banking_transaction_analysis.py     # writes everything to outputs/
```
In Google Colab: upload the notebook, run all cells, and choose `banking_transactions.csv` when
the upload dialog (step 2) appears.

## Using a real dataset (e.g. Kaggle)
Keep the columns `TxnID, AccountID, TxnDate, Type, Amount, MerchantCategory, BalanceAfter`.
If the file has no `IsFraud` column, the script creates a rule-based proxy label and prints a warning.
All random steps use `random_state = 42`.
