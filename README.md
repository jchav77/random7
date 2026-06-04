
Project context

We ran a controlled test on repo timing for an auto loan portfolio. Accounts were randomly assigned (RNG) to:

	•	tst_ind = 0 (BAU): eligible for repo assignment at min 74 DPD
	•	tst_ind = 1 (test): held until min 84/85 DPD

Outcome is if_charge_off (0/1). The question isn’t “did the policy work on average” — it’s “which natural behavioral segments are affected differently by the 10-day hold.” You already know the columns from the dataset build.

Methodology — two stages. Stage one discovers segments, stage two measures policy impact within each.

	1.	EDA — profile the data: row count, tst_ind split, overall charge-off rate, distributions, missing values, outliers, correlation matrix.
	2.	Feature prep — engineer useful features (e.g. days_since_last_payment), log-transform skewed ones, standardize with StandardScaler (save it). Run a throwaway Random Forest on if_charge_off ONLY to rank feature importance and drop near-zero-signal features. Don’t reuse that model.
	3.	KMeans — cluster on behavioral features ONLY. Exclude if_charge_off, tst_ind, dpd, eligibility/distribution date, and all ID/date fields — clustering is on who the customer is, not the outcome or the policy lever. Pick K via elbow + silhouette. Profile and name each cluster.
	4.	Outcome overlay — charge-off rate per cluster (ignore tst_ind here). Worst-behaving cluster should have the highest rate.
	5.	Decision tree — target = cluster label, NOT charge-off. Makes segments reproducible. max_depth ~5, min_samples_leaf ~100, 80/20 stratified, 5-fold CV. Report accuracy, per-cluster F1, confusion matrix. Run Random Forest + XGBoost on the same target as robustness checks.
	6.	Policy impact per cluster — within each cluster compare charge-off rate for tst_ind=0 vs 1. Chi-square, falling back to Fisher’s exact if any expected cell < 5. Report n, rates, delta, p-value, test used.
	7.	Recommendation table — per cluster: size, % of portfolio, rate at 74, rate at 84/85, delta, significance, recommendation (move to 85 / hold at 74).

Warning flags — print WARNINGS, don’t halt:

	•	tst_ind split outside ~40/60
	•	overall charge-off rate under ~5%
	•	any two features correlated above 0.85
	•	any cluster under ~2% or over ~60% of portfolio
	•	silhouette below 0.25
	•	any cluster lacking a reasonable mix of tst_ind 0 and 1
	•	any cluster with under ~500 accounts in either timing group
	•	whenever significance test falls back to Fisher’s

Requirements: clean modular Python (function per step), scikit-learn / pandas / numpy / scipy / xgboost. Print a short summary + any warnings at each step so I can eyeball the run.

Confirm you understand the two-stage logic, then start with Step 1.

