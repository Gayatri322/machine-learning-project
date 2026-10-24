# =====================================================================
#  reproduce.py - one command that reproduces every result in the project
#  Run:  python reproduce.py
#  Needs: pip install -r requirements.txt
#  Run it from the repository folder (the one that contains this file).
# =====================================================================
import subprocess, sys, time

STEPS = [
    ("1/4  Generate the datasets (data/*.csv)",         "generate_data.py"),
    ("2/4  Train and compare the six models",           "train_models.py"),
    ("3/4  Extrapolation test (train W/h<7, test 8-10)", "extrapolation_test.py"),
    ("4/4  Draw the figure used in the report",         "make_report_figure.py"),
]

for title, script in STEPS:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
    t0 = time.time()
    # run each script with the same Python that is running this file
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\nSTOPPED: {script} failed. Read the error message above.")
        sys.exit(result.returncode)
    print(f"-- finished {script} in {time.time() - t0:.0f} s")

print("\nAll done. New files: results_*.csv, errors_*.png, "
      "predicted_vs_actual_*.png, curve_microstrip.png, fig_curves.png")
