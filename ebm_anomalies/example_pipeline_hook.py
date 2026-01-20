
# Example hook to run SHAP explainability after VAE scoring

from ebm_anomalies.shap_explainability import ShapConfig, generate_shap_for_anomalies

cfg = ShapConfig(
    output_dir="outputs/shap",
    max_anomalies=100,
    background_sample=2000,
    surrogate_model_path="artifacts/shap_surrogate_lgbm.pkl"
)

result = generate_shap_for_anomalies(
    df_scored=df_scored,                 # scored dataframe
    feature_cols=FEATURE_COLUMNS,        # final 287 model features
    score_col="reconstruction_error",
    anomaly_flag_col="is_anomaly",
    id_cols=["taxpayer_tin", "tax_period"],
    cfg=cfg
)

print(result)
