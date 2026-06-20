from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import uvicorn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

app = FastAPI(
    title="API Rétention Client",
    description="""
    API REST pour la prédiction du churn client et l'estimation du revenu à risque.

    ## Endpoints disponibles
    - **GET /health** → Vérifie que l'API est active
    - **POST /predict** → Prédit le churn et le revenu à risque
    - **GET /model-info** → Informations sur les modèles
    """,
    version="1.0.0"
)

# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================
print("⏳ Chargement des modèles...")

try:
    preprocessor = joblib.load('../models/preprocessor.pkl')
    rf_clf       = joblib.load('../models/rf_classification.pkl')
    xgb_reg      = joblib.load('../models/xgb_regression.pkl')

    X_test     = np.load('../data/processed/X_test.npy')
    y_test_clf = np.load('../data/processed/y_test_clf.npy')
    y_test_reg = np.load('../data/processed/y_test_reg.npy')

    y_proba_clf = rf_clf.predict_proba(X_test)[:, 1]
    y_pred_clf  = (y_proba_clf >= 0.45).astype(int)

    RF_METRICS = {
        "ROC-AUC"  : round(float(roc_auc_score(y_test_clf, y_proba_clf)), 4),
        "Recall"   : round(float(recall_score(y_test_clf, y_pred_clf)), 4),
        "Precision": round(float(precision_score(y_test_clf, y_pred_clf)), 4),
        "F1-Score" : round(float(f1_score(y_test_clf, y_pred_clf)), 4),
        "Accuracy" : round(float(accuracy_score(y_test_clf, y_pred_clf)), 4),
    }

    churners_mask = y_test_reg > 0
    y_pred_reg    = np.maximum(xgb_reg.predict(X_test[churners_mask]).flatten(), 0)

    XGB_METRICS = {
        "MAE" : round(float(mean_absolute_error(y_test_reg[churners_mask], y_pred_reg)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test_reg[churners_mask], y_pred_reg))), 2),
        "R2"  : round(float(r2_score(y_test_reg[churners_mask], y_pred_reg)), 4),
    }

    print("✅ Modèles chargés !")
    print(f"   RF  — ROC-AUC={RF_METRICS['ROC-AUC']} | Recall={RF_METRICS['Recall']}")
    print(f"   XGB — MAE={XGB_METRICS['MAE']}€ | R²={XGB_METRICS['R2']}")
    MODELS_LOADED = True

except Exception as e:
    print(f"❌ Erreur : {e}")
    MODELS_LOADED = False
    RF_METRICS  = {}
    XGB_METRICS = {}

# ============================================================
# SCHÉMA DE DONNÉES
# ============================================================
class ClientData(BaseModel):
    gender: str = Field(..., example="Male")
    age: int = Field(..., ge=18, le=74, example=35)
    country: str = Field(..., example="Germany")
    customer_segment: str = Field(..., example="Individual")
    tenure_months: int = Field(..., ge=1, le=59, example=12)
    signup_channel: str = Field(..., example="Web")
    contract_type: str = Field(..., example="Monthly")
    monthly_logins: int = Field(..., ge=0, le=54, example=15)
    weekly_active_days: int = Field(..., ge=0, le=7, example=3)
    avg_session_time: float = Field(..., ge=1.0, le=42.0, example=15.0)
    features_used: int = Field(..., ge=1, le=15, example=5)
    usage_growth_rate: float = Field(..., ge=-0.58, le=0.54, example=0.02)
    last_login_days_ago: int = Field(..., ge=0, le=80, example=5)
    monthly_fee: int = Field(..., ge=10, le=100, example=30)
    payment_method: str = Field(..., example="Card")
    payment_failures: int = Field(..., ge=0, le=5, example=0)
    discount_applied: str = Field(..., example="No")
    price_increase_last_3m: str = Field(..., example="No")
    support_tickets: int = Field(..., ge=0, le=7, example=1)
    avg_resolution_time: float = Field(..., ge=1.0, le=62.0, example=24.0)
    complaint_type: str = Field(..., example="No_Complaint")
    csat_score: float = Field(..., ge=1.0, le=5.0, example=3.5)
    escalations: int = Field(..., ge=0, le=4, example=0)
    email_open_rate: float = Field(..., ge=0.1, le=0.9, example=0.5)
    marketing_click_rate: float = Field(..., ge=0.01, le=0.5, example=0.25)
    nps_score: int = Field(..., ge=-100, le=100, example=20)
    survey_response: str = Field(..., example="Satisfied")
    referral_count: int = Field(..., ge=0, le=7, example=1)

# ============================================================
# ENDPOINT 1 — GET /health
# ============================================================
@app.get("/health", tags=["Santé"])
def health_check():
    return {
        "status": "ok" if MODELS_LOADED else "degraded",
        "models_loaded": MODELS_LOADED,
        "models": {
            "classification": "Random Forest ✅" if MODELS_LOADED else "❌",
            "regression"    : "XGBoost ✅" if MODELS_LOADED else "❌",
            "preprocessor"  : "StandardScaler + OneHotEncoder ✅" if MODELS_LOADED else "❌"
        },
        "version": "1.0.0"
    }

# ============================================================
# ENDPOINT 2 — POST /predict
# ============================================================
@app.post("/predict", tags=["Prédiction"])
def predict(client: ClientData):
    """
    Prédit la probabilité de churn et le revenu à risque.

    - Classification : **Random Forest**
    - Régression : **XGBoost**
    """
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Modèles non chargés.")

    try:
        client_dict = client.model_dump()
        df_client   = pd.DataFrame([client_dict])

        # Supprimer total_revenue car exclu du preprocessing
        df_client = df_client.drop(columns=['total_revenue'], errors='ignore')

        X_client = preprocessor.transform(df_client)

        # Classification — Random Forest
        proba_churn = float(rf_clf.predict_proba(X_client)[0, 1])
        churn_pred  = int(proba_churn >= 0.45)

        if proba_churn >= 0.45:
            risk_level     = "élevé"
            recommendation = "Action immédiate requise — Contacter ce client et proposer une offre de fidélisation"
        elif proba_churn >= 0.3:
            risk_level     = "modéré"
            recommendation = "Surveillance recommandée — Planifier un appel de satisfaction sous 7 jours"
        else:
            risk_level     = "faible"
            recommendation = "Client fidèle — Maintenir l'engagement et explorer les opportunités d'upsell"

        # Régression XGBoost — prédiction directe du revenue_at_risk
        if churn_pred == 1:
            revenue_at_risk = round(float(np.maximum(xgb_reg.predict(X_client), 0)[0]), 2)
        else:
            revenue_at_risk = 0.0

        return {
            "status": "success",
            "input_summary": {
                "segment"      : client.customer_segment,
                "tenure_months": client.tenure_months,
                "contract_type": client.contract_type,
                "monthly_fee"  : client.monthly_fee,
            },
            "classification": {
                "proba_churn"     : round(proba_churn, 4),
                "churn_prediction": churn_pred,
                "risk_level"      : risk_level,
                "model_used"      : f"Random Forest (ROC-AUC = {RF_METRICS.get('ROC-AUC', 'N/A')})"
            },
            "regression": {
                "revenue_at_risk": revenue_at_risk,
                "model_used"     : f"XGBoost (MAE = {XGB_METRICS.get('MAE', 'N/A')}€)"
            },
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur : {str(e)}")

# ============================================================
# LANCEMENT
# ============================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)