# Machine learning surrogates for microstrip line and patch antenna

Predicting microstrip line impedance (Z0) and patch antenna resonant
frequency (fr) from geometry, using six regression models.
Method follows Balaji and Raghavan, arXiv:2406.04357.

## Contents
- generate_data.py - creates the synthetic datasets from analytical equations
- train_models.py  - trains and compares the six models
- data/            - train, validation and test CSV files (70/15/15 split)
- results/         - error tables and plots
- poster/          - project proposal poster (PDF)

## Data
Synthetic, 10,000 samples per problem, generated from the equations of
Gupta and Garg (microstrip Z0) and Singh (patch fr).

## Run
pip install numpy pandas scikit-learn matplotlib
python generate_data.py   (optional, recreates the data)
python train_models.py

## Team
Madhesh S, Harsh Prabha Bhaina, Gayatri Supraja
