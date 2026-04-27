# ============================================================
# API REST — FastAPI
# Rétention Client & Risque de Revenus
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import uvicorn

# ============================================================
# INITIALISATION DE L'APP
# ============================================================
app = FastAPI(
    title="API Rétention Client",
    description="""
    API REST pour la prédiction du churn client et l'estimation
    du revenu à risque.
    
    ## Endpoints disponibles
    - **GET /health** → Vérifie que l'API est active
    - **POST /predict** → Prédit le churn et le revenu à risque
    - **GET /model-info** → Informations sur les modèles
    """,
    version="1.0.0"
)

# ============================================================
# CHARGEMENT DES MODÈLES AU DÉMARRAGE
# ============================================================
print("⏳ Chargement des modèles...")

try:
    preprocessor = joblib.load('../models/preprocessor.pkl')
    rf_clf  = joblib.load('../models/rf_classification.pkl')
    mlp_reg = keras.models.load_model('../models/mlp_regression.keras')
    print("✅ Modèles chargés avec succès !")
    MODELS_LOADED = True
except Exception as e:
    print(f"❌ Erreur chargement modèles : {e}")
    MODELS_LOADED = False

# ============================================================
# SCHÉMA DE DONNÉES — Ce que l'API attend en entrée
# ============================================================
class ClientData(BaseModel):
    # Profil client
    gender: str = Field(..., example="Male",
                        description="Genre : Male ou Female")
    age: int = Field(..., ge=18, le=74,
                     example=35, description="Âge du client (18-74)")
    country: str = Field(..., example="France",
                         description="Pays du client")
    customer_segment: str = Field(..., example="Individual",
                                   description="Individual, SME ou Enterprise")
    tenure_months: int = Field(..., ge=1, le=59,
                                example=12,
                                description="Ancienneté en mois")
    signup_channel: str = Field(..., example="Web",
                                 description="Web, Mobile ou Referral")
    contract_type: str = Field(..., example="Monthly",
                                description="Monthly, Quarterly ou Yearly")

    # Utilisation
    monthly_logins: int = Field(..., ge=0, le=54, example=15)
    weekly_active_days: int = Field(..., ge=0, le=7, example=3)
    avg_session_time: float = Field(..., ge=1.0, le=42.0, example=15.0)
    features_used: int = Field(..., ge=1, le=15, example=5)
    usage_growth_rate: float = Field(..., ge=-0.58, le=0.54, example=0.02)
    last_login_days_ago: int = Field(..., ge=0, le=80, example=5)

    # Facturation
    monthly_fee: int = Field(..., ge=10, le=100, example=30)
    total_revenue: int = Field(..., ge=10, le=5900, example=500)
    payment_method: str = Field(..., example="Card",
                                 description="Card, PayPal ou Bank Transfer")
    payment_failures: int = Field(..., ge=0, le=5, example=0)
    discount_applied: str = Field(..., example="No",
                                   description="Yes ou No")
    price_increase_last_3m: str = Field(..., example="No",
                                         description="Yes ou No")

    # Support
    support_tickets: int = Field(..., ge=0, le=7, example=1)
    avg_resolution_time: float = Field(..., ge=1.0, le=62.0, example=24.0)
    complaint_type: str = Field(..., example="No_Complaint",
                                 description="No_Complaint, Technical, Billing ou Service")
    csat_score: float = Field(..., ge=1.0, le=5.0, example=3.5)
    escalations: int = Field(..., ge=0, le=4, example=0)

    # Engagement
    email_open_rate: float = Field(..., ge=0.1, le=0.9, example=0.5)
    marketing_click_rate: float = Field(..., ge=0.01, le=0.5, example=0.25)
    nps_score: int = Field(..., ge=-100, le=100, example=20)
    survey_response: str = Field(..., example="Satisfied",
                                  description="Satisfied, Neutral ou Unsatisfied")
    referral_count: int = Field(..., ge=0, le=7, example=1)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "age": 35,
                "country": "France",
                "customer_segment": "Individual",
                "tenure_months": 12,
                "signup_channel": "Web",
                "contract_type": "Monthly",
                "monthly_logins": 15,
                "weekly_active_days": 3,
                "avg_session_time": 15.0,
                "features_used": 5,
                "usage_growth_rate": 0.02,
                "last_login_days_ago": 5,
                "monthly_fee": 30,
                "total_revenue": 500,
                "payment_method": "Card",
                "payment_failures": 0,
                "discount_applied": "No",
                "price_increase_last_3m": "No",
                "support_tickets": 1,
                "avg_resolution_time": 24.0,
                "complaint_type": "No_Complaint",
                "csat_score": 3.5,
                "escalations": 0,
                "email_open_rate": 0.5,
                "marketing_click_rate": 0.25,
                "nps_score": 20,
                "survey_response": "Satisfied",
                "referral_count": 1
            }
        }

# ============================================================
# ENDPOINT 1 — GET /health
# ============================================================
@app.get("/health", tags=["Santé"])
def health_check():
    """
    Vérifie que l'API est active et que les modèles sont chargés.
    """
    return {
        "status": "ok" if MODELS_LOADED else "degraded",
        "models_loaded": MODELS_LOADED,
        "models": {
            "classification": "Random Forest ✅" if MODELS_LOADED else "❌",
            "regression": "MLP ✅" if MODELS_LOADED else "❌",
            "preprocessor": "StandardScaler + OneHotEncoder ✅" if MODELS_LOADED else "❌"
        },
        "version": "1.0.0"
    }

# ============================================================
# ENDPOINT 2 — POST /predict
# ============================================================
@app.post("/predict", tags=["Prédiction"])
def predict(client: ClientData):
    """
    Prédit la probabilité de churn et le revenu à risque
    pour un client donné.

    **Modèles utilisés :**
    - Classification : **Random Forest** (ROC-AUC = 0.799)
    - Régression : **MLP Deep Learning** (MAE = 138.85€)

    **Retourne :**
    - `proba_churn` : probabilité de résiliation (0 à 1)
    - `churn_prediction` : 0 (reste) ou 1 (part)
    - `risk_level` : faible / modéré / élevé
    - `revenue_at_risk` : revenu estimé à risque en € (MLP)
    - `recommendation` : action recommandée
    """

    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Les modèles ne sont pas chargés. Vérifiez les fichiers."
        )

    try:
        # Convertir en DataFrame
        client_dict = client.model_dump()
        df_client = pd.DataFrame([client_dict])

        # Preprocessing — même pipeline que l'entraînement
        X_client = preprocessor.transform(df_client)

        # ============================================================
        # TÂCHE 1 — CLASSIFICATION (Random Forest)
        # ============================================================
        proba_churn = float(rf_clf.predict_proba(X_client)[0, 1])
        churn_pred  = int(proba_churn >= 0.5)

        # Niveau de risque
        if proba_churn >= 0.5:
            risk_level = "élevé"
            recommendation = (
                "Action immédiate requise — "
                "Contacter ce client et proposer une offre de fidélisation"
            )
        elif proba_churn >= 0.3:
            risk_level = "modéré"
            recommendation = (
                "Surveillance recommandée — "
                "Planifier un appel de satisfaction sous 7 jours"
            )
        else:
            risk_level = "faible"
            recommendation = (
                "Client en bonne santé — "
                "Maintenir l'engagement et explorer les opportunités d'upsell"
            )

        # ============================================================
        # TÂCHE 2 — RÉGRESSION (MLP Deep Learning)
        # ============================================================
        revenue_at_risk_raw = mlp_reg.predict(X_client)[0][0]
        revenue_at_risk = max(round(float(revenue_at_risk_raw), 2), 0.0)

        return {
            "status": "success",
            "input_summary": {
                "segment": client.customer_segment,
                "tenure_months": client.tenure_months,
                "contract_type": client.contract_type,
                "monthly_fee": client.monthly_fee,
                "total_revenue": client.total_revenue
            },
            "classification": {
                "proba_churn": round(proba_churn, 4),
                "churn_prediction": churn_pred,
                "risk_level": risk_level,
                "model_used": "Random Forest (ROC-AUC = 0.799)"
            },
            "regression": {
                "revenue_at_risk": revenue_at_risk,
                "total_revenue": client.total_revenue,
                "model_used": "MLP Deep Learning (MAE = 138.85€)"
            },
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )

# ============================================================
# ENDPOINT 3 — GET /model-info
# ============================================================
@app.get("/model-info", tags=["Informations"])
def model_info():
    """
    Retourne les informations sur les modèles utilisés.
    """
    return {
        "classification": {
            "name": "Random Forest Classifier",
            "type": "Ensemble — Arbres de décision",
            "why_chosen": (
                "Meilleur ROC-AUC (0.799) et F1-Score (0.361) "
                "parmi les 4 modèles testés. "
                "Robuste aux outliers et fournit une feature importance native."
            ),
            "metrics": {
                "ROC-AUC": 0.7993,
                "F1-Score": 0.3612,
                "Recall": 0.4559,
                "Precision": 0.2990,
                "Accuracy": 0.8355
            },
            "parameters": {
                "n_estimators": 200,
                "max_depth": 10,
                "class_weight": "balanced",
                "min_samples_leaf": 5
            }
        },
        "regression": {
            "name": "MLP Regressor (Deep Learning)",
            "type": "Réseau de neurones multicouches",
            "why_chosen": (
                "Meilleure MAE (138.85€) et R² (0.070) "
                "parmi les 4 modèles testés. "
                "Capture les relations non-linéaires de revenue_at_risk."
            ),
            "metrics": {
                "MAE": 138.85,
                "RMSE": 371.82,
                "R2": 0.0700
            },
            "architecture": {
                "layers": [
                    "Input(51)",
                    "Dense(128) + ReLU + Dropout(0.3)",
                    "Dense(64) + ReLU + Dropout(0.2)",
                    "Dense(32) + ReLU",
                    "Dense(1) + Linear"
                ],
                "optimizer": "Adam (lr=0.001)",
                "loss": "MSE"
            }
        },
        "preprocessing": {
            "scaler": "StandardScaler sur 19 variables numériques",
            "encoder": "OneHotEncoder sur 10 variables catégorielles",
            "total_features": 51,
            "train_size": 8000,
            "test_size": 2000,
            "split_strategy": "Stratified 80/20"
        },
        "dataset": {
            "source": "Kaggle — Customer Churn Business Dataset",
            "n_clients": 10000,
            "churn_rate": "10.2%",
            "revenue_at_risk_total": "862,640 €"
        }
    }

# ============================================================
# LANCEMENT DE L'API
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )