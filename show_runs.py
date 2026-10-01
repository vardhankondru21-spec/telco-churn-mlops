import mlflow

client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('Telco_Churn_Prediction')
runs = client.search_runs(exp.experiment_id, order_by=['start_time DESC'])

for r in runs:
    print(f"Run: {r.info.run_name} | Status: {r.info.status}")
    for k, v in r.data.metrics.items():
        print(f"    {k}: {v:.4f}")
    print()