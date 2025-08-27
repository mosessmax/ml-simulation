# ml simulation

ml tool just with learning curves and some dataset personalities to play with.

## what it does

- linear regression from scratch (no shortcuts)
- watch your model learn in real-time with loss curves
- 4 different datasets to test on:
  - clean linear (easy mode) 
  - noisy sales data (realistic)  
  - housing prices (practical) 
  - polynomial disguised (tricky) 
- interactive predictions after training
- load your own csv files too

## quick start

```bash
# setup with uv
uv init
uv add numpy pandas matplotlib seaborn scikit-learn

# or just 
uv sync

# run it
uv run python3 main.py
```

## how to use

1. run the script
2. choose a dataset (or load your csv)
3. watch the training happen
4. see the results and plots
5. test predictions interactively
6. type 'q' when done

## sample data included

- `data/housing_sample.csv` - house prices (sqft, bedrooms, age → price)
- `data/sales_sample.csv` - business revenue (marketing, team, quarter → revenue)

both have realistic relationships and some noise. good for testing.

## the vibe

keeping it simple and focused. no fancy frameworks, just numpy and matplotlib. you probably have seen this somewhere, just that this one has visual feedback.

perfect for understanding how linear regression actually works under the hood.