# Machine learning surrogates for microstrip line and patch antenna

Predicting microstrip line impedance (Z0) and patch antenna resonant
frequency (fr) from geometry, using six regression models.
Method follows Balaji and Raghavan, arXiv:2406.04357.

## Contents
- generate_data.py - creates the synthetic datasets from analytical equations
- train_models.py  - trains and compares the six models
- extrapolation_test.py - trains on W/h < 7, tests on W/h 8-10
- make_report_figure.py - draws the figure used in the report
- requirements.txt - Python libraries needed
- data/            - train, validation and test CSV files (70/15/15 split)
- results/         - error tables and plots
- poster/          - project proposal poster (PDF)
- report/          - CVPR-style report (PDF and LaTeX source)

## Data
Synthetic, 10,000 samples per problem, generated from the equations of
Gupta and Garg (microstrip Z0) and Singh (patch fr).

## Reproduce the results
1. pip install -r requirements.txt
2. python generate_data.py          (recreates the 6 CSV files in data/; fixed seed 42)
3. python train_models.py           (Tables: results_*.csv and plots)
4. python extrapolation_test.py     (extrapolation results)
5. python make_report_figure.py     (figure for the report)

Run all commands from the repository folder. Results can differ slightly
between library versions, but the model ranking stays the same.

## Team
Madhesh S, Harsh Prabha Bhaina, Gayatri Supraja K
