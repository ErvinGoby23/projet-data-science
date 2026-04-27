# ============================================================
# DASHBOARD STREAMLIT — Rétention Client & Risque de Revenus
# Données issues des notebooks de modélisation
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
import os
from tensorflow import keras

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Rétention Client — Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CHARGEMENT DES MODÈLES ET DONNÉES
# ============================================================
@st.cache_resource
def load_models():
    rf_clf  = joblib.load('../models/rf_classification.pkl')
    mlp_reg = keras.models.load_model('../models/mlp_regression.keras')
    preprocessor = joblib.load('../models/preprocessor.pkl')
    return rf_clf, mlp_reg, preprocessor

@st.cache_data
def load_data():
    df = pd.read_csv('../data/customer_churn_business_dataset.csv')
    return df

rf_clf, mlp_reg, preprocessor = load_models()
df = load_data()

# ============================================================
# SIDEBAR — NAVIGATION
# ============================================================
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Choisir une page",
    [
        "🏠 Vue d'ensemble",
        "📈 Analyse des données",
        "🤖 Comparaison des modèles",
        "🔮 Prédiction client",
        "🔍 Interprétabilité SHAP"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Projet Data Science M1**")
st.sidebar.markdown("Rétention Client & Risque de Revenus")
st.sidebar.markdown("Dataset : 10 000 clients SaaS")

# ============================================================
# PAGE 1 — VUE D'ENSEMBLE
# KPIs issus de l'EDA (EDA.ipynb) :
# - 10 000 clients, 10.2% de churners, revenu à risque 862 640€
# ============================================================
if page == "🏠 Vue d'ensemble":

    st.title("📊 Système Intelligent de Rétention Client")
    st.markdown("### Tableau de bord décisionnel — Vue d'ensemble")
    st.markdown("---")

    # KPIs — valeurs issues de l'EDA
    total_clients  = len(df)                          # 10 000
    churners       = df['churn'].sum()                # 1021
    taux_churn     = df['churn'].mean() * 100         # 10.2%
    revenu_risque  = df[df['churn']==1]['total_revenue'].sum()  # 862 640€
    revenu_moyen   = df['total_revenue'].mean()       # 1057€

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="👥 Total Clients",
            value=f"{total_clients:,}"
        )
    with col2:
        st.metric(
            label="⚠️ Clients à Risque",
            value=f"{churners:,}",
            delta=f"{taux_churn:.1f}% du total",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            label="💰 Revenu à Risque",
            value=f"{revenu_risque:,.0f} €",
            delta_color="inverse"
        )
    with col4:
        st.metric(
            label="📈 Revenu Moyen/Client",
            value=f"{revenu_moyen:,.0f} €"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Distribution churn — issue de l'EDA
        # Résultat : 8979 restent (89.8%) vs 1021 partent (10.2%)
        st.subheader("Distribution du Churn")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df['churn'].value_counts()
        ax.pie(
            counts.values,
            labels=['Reste (0)', 'Part (1)'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax.set_title('Proportion Churn / Non-Churn\n(89.8% vs 10.2%)')
        st.pyplot(fig)
        plt.close()

    with col2:
        # Churn par segment — issue de l'EDA
        # SME : 30.9%, Individual : 9.4%, Enterprise : 10.2%
        st.subheader("Churn par Segment Client")
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_seg = df.groupby('customer_segment')['churn'].mean() * 100
        bars = ax.bar(
            churn_seg.index,
            churn_seg.values,
            color=['#3498db', '#e74c3c', '#2ecc71']
        )
        ax.axhline(
            y=taux_churn, color='black',
            linestyle='--', label=f'Moyenne ({taux_churn:.1f}%)'
        )
        ax.set_ylabel('Taux de churn (%)')
        ax.set_title('Taux de churn par segment\n(SME = segment le plus à risque)')
        ax.legend()
        for bar, val in zip(bars, churn_seg.values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontweight='bold'
            )
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Top 10 clients à risque — calculé en temps réel
    st.subheader("🔴 Top 10 Clients les Plus à Risque")

    df_clean = df.copy()
    df_clean['complaint_type'] = df_clean['complaint_type'].fillna('No_Complaint')
    df_feat = df_clean.drop(columns=['customer_id', 'city', 'churn'], errors='ignore')

    X_all   = preprocessor.transform(df_feat)
    probas  = rf_clf.predict_proba(X_all)[:, 1]

    df_risk = df.copy()
    df_risk['proba_churn']    = probas
    df_risk['revenue_at_risk'] = df_risk['total_revenue'] * df_risk['proba_churn']

    top10 = df_risk.nlargest(10, 'revenue_at_risk')[[
        'customer_id', 'customer_segment', 'contract_type',
        'total_revenue', 'proba_churn', 'revenue_at_risk'
    ]].reset_index(drop=True)

    top10['proba_churn']    = top10['proba_churn'].apply(lambda x: f"{x:.1%}")
    top10['revenue_at_risk'] = top10['revenue_at_risk'].apply(lambda x: f"{x:,.0f} €")
    top10['total_revenue']   = top10['total_revenue'].apply(lambda x: f"{x:,.0f} €")

    st.dataframe(top10, use_container_width=True)

# ============================================================
# PAGE 2 — ANALYSE DES DONNÉES
# Graphiques issus de l'EDA (EDA.ipynb)
# ============================================================
elif page == "📈 Analyse des données":

    st.title("📈 Analyse Exploratoire des Données")
    st.markdown("Résultats issus de l'EDA réalisée sur le dataset de 10 000 clients.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Issue de l'EDA — Australia 11.5% vs India 9.2%
        st.subheader("Churn par Pays")
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_country = df.groupby('country')['churn'].mean() * 100
        churn_country.sort_values().plot(kind='barh', ax=ax, color='#3498db')
        ax.axvline(
            x=df['churn'].mean()*100, color='red',
            linestyle='--', label='Moyenne (10.2%)'
        )
        ax.set_xlabel('Taux de churn (%)')
        ax.set_title('Australia = pays le plus à risque (11.5%)')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        # Issue de l'EDA — Monthly dominant (50%), Yearly le plus fidèle
        st.subheader("Churn par Type de Contrat")
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_contract = df.groupby('contract_type')['churn'].mean() * 100
        bars = ax.bar(
            churn_contract.index,
            churn_contract.values,
            color='#9b59b6'
        )
        ax.axhline(
            y=df['churn'].mean()*100, color='red',
            linestyle='--', label='Moyenne (10.2%)'
        )
        ax.set_ylabel('Taux de churn (%)')
        ax.tick_params(axis='x', rotation=0)
        ax.legend()
        for bar, val in zip(bars, churn_contract.values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{val:.1f}%', ha='center', fontweight='bold'
            )
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Issue de l'EDA — nouveaux clients (tenure bas) partent plus
        st.subheader("Ancienneté vs Revenu Total")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = df['churn'].map({0: '#2ecc71', 1: '#e74c3c'})
        ax.scatter(
            df['tenure_months'], df['total_revenue'],
            c=colors, alpha=0.3, s=10
        )
        ax.set_xlabel('Ancienneté (mois)')
        ax.set_ylabel('Revenu Total (€)')
        ax.set_title('Rouge = churners, Vert = fidèles\n(Nouveaux clients plus à risque)')
        st.pyplot(fig)
        plt.close()

    with col2:
        # Issue de l'EDA — NPS bas = risque churn
        # Corrélation NPS / churn = -0.01 (faible mais visible)
        st.subheader("NPS Score vs Churn")
        fig, ax = plt.subplots(figsize=(6, 4))
        df.boxplot(column='nps_score', by='churn', ax=ax, patch_artist=True)
        ax.set_xlabel('Churn (0=Non, 1=Oui)')
        ax.set_ylabel('NPS Score')
        ax.set_title('Distribution NPS par groupe churn')
        plt.suptitle('')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Heatmap — issue de l'EDA
    # Corrélations clés : csat_score (-0.16), tenure_months (-0.12),
    # payment_failures (+0.11), monthly_fee/total_revenue (0.71)
    st.subheader("🔥 Heatmap des Corrélations Clés")
    st.caption(
        "Corrélations clés : csat_score (-0.16), "
        "tenure_months (-0.12), payment_failures (+0.11), "
        "monthly_fee ↔ total_revenue (0.71)"
    )

    cols_corr = [
        'age', 'tenure_months', 'monthly_fee',
        'total_revenue', 'payment_failures',
        'support_tickets', 'csat_score',
        'nps_score', 'churn'
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    corr = df[cols_corr].corr()
    sns.heatmap(
        corr, annot=True, fmt='.2f',
        cmap='RdYlGn', center=0, ax=ax,
        linewidths=0.5, annot_kws={'size': 9}
    )
    ax.set_title('Corrélations entre variables clés', fontsize=13)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("💡 Signaux Forts Détectés en EDA")

    # Calcul en temps réel sur les 10 000 clients
    churn_price    = df[df['price_increase_last_3m']=='Yes']['churn'].mean()*100
    churn_sme      = df[df['customer_segment']=='SME']['churn'].mean()*100
    churn_mobile   = df[df['signup_channel']=='Mobile']['churn'].mean()*100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(f"""
        **Hausse de prix récente**
        Taux de churn : **{churn_price:.1f}%**
        vs moyenne {df['churn'].mean()*100:.1f}%
        """)
    with col2:
        st.error(f"""
        **Segment SME**
        Taux de churn : **{churn_sme:.1f}%**
        vs moyenne {df['churn'].mean()*100:.1f}%
        """)
    with col3:
        st.error(f"""
        **Canal Mobile**
        Taux de churn : **{churn_mobile:.1f}%**
        vs moyenne {df['churn'].mean()*100:.1f}%
        """)

# ============================================================
# PAGE 3 — COMPARAISON DES MODÈLES
# Métriques issues des notebooks de modélisation :
# - LR : notebook 03
# - RF : notebook 04
# - XGB : notebook 05
# - MLP : notebook 06
# ============================================================
elif page == "🤖 Comparaison des modèles":

    st.title("🤖 Comparaison des 4 Modèles")
    st.markdown("Métriques calculées sur le jeu de test (2000 clients, 10.2% churners).")
    st.markdown("---")

    # ============================================================
    # CLASSIFICATION — valeurs issues des notebooks
    # LR  : Accuracy=0.6685, Precision=0.1834, Recall=0.6520,
    #        F1=0.2863, ROC-AUC=0.7212
    # RF  : Accuracy=0.8355, Precision=0.2990, Recall=0.4559,
    #        F1=0.3612, ROC-AUC=0.7993
    # XGB : Accuracy=0.8615, Precision=0.2583, Recall=0.1912,
    #        F1=0.2197, ROC-AUC=0.7570
    # MLP : Accuracy=0.7180, Precision=0.2068, Recall=0.6225,
    #        F1=0.3105, ROC-AUC=0.7373
    # ============================================================
    results_data = {
        'Modèle'   : ['Logistic Regression', 'Random Forest',
                      'XGBoost', 'MLP (Deep Learning)'],
        'Accuracy' : [0.6685, 0.8355, 0.8615, 0.7180],
        'Precision': [0.1834, 0.2990, 0.2583, 0.2068],
        'Recall'   : [0.6520, 0.4559, 0.1912, 0.6225],
        'F1-Score' : [0.2863, 0.3612, 0.2197, 0.3105],
        'ROC-AUC'  : [0.7212, 0.7993, 0.7570, 0.7373]
    }
    df_results = pd.DataFrame(results_data)

    st.subheader("📊 Tableau Comparatif — Classification")
    st.dataframe(
        df_results.style.highlight_max(
            subset=['Accuracy', 'Precision', 'Recall',
                    'F1-Score', 'ROC-AUC'],
            color='#d4edda'
        ),
        use_container_width=True
    )

    st.markdown("---")

    # Graphique comparatif
    metriques = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['#3498db', '#27ae60', '#e67e22', '#9b59b6']

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, metric in enumerate(metriques):
        bars = axes[i].bar(
            ['LR', 'RF', 'XGB', 'MLP'],
            df_results[metric],
            color=colors, alpha=0.85
        )
        axes[i].set_title(metric, fontweight='bold')
        axes[i].set_ylim(0, 1)
        for bar, val in zip(bars, df_results[metric]):
            axes[i].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=8
            )
    plt.suptitle('Comparaison des 4 modèles', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ============================================================
    # RÉGRESSION — valeurs issues des notebooks
    # Ridge : MAE=156.11, RMSE=375.15, R²=0.0533
    # RF    : MAE=145.39, RMSE=373.57, R²=0.0612
    # XGB   : MAE=158.61, RMSE=386.03, R²=-0.0024
    # MLP   : MAE=138.85, RMSE=371.82, R²=0.0700
    # ============================================================
    st.subheader("📊 Tableau Comparatif — Régression (Revenue at Risk)")
    reg_data = {
        'Modèle'  : ['Ridge Regression', 'Random Forest',
                     'XGBoost', 'MLP (Deep Learning)'],
        'MAE (€)' : [156.11, 145.39, 158.61, 138.85],
        'RMSE (€)': [375.15, 373.57, 386.03, 371.82],
        'R²'      : [0.0533, 0.0612, -0.0024, 0.0700]
    }
    df_reg = pd.DataFrame(reg_data)
    st.dataframe(
        df_reg.style.highlight_min(
            subset=['MAE (€)', 'RMSE (€)'], color='#d4edda'
        ).highlight_max(
            subset=['R²'], color='#d4edda'
        ),
        use_container_width=True
    )

    st.markdown("---")

    # Matrice de confusion RF — issue du notebook 04
    # TN=1578, FP=218, FN=111, TP=93
    st.subheader("🔴 Analyse des Erreurs — Random Forest (modèle final)")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Matrice de confusion — Random Forest**
        - Vrais Négatifs  (TN) : **1578** ✅
        - Faux Positifs   (FP) : **218**
        - Faux Négatifs   (FN) : **111** ⚠️ churners manqués
        - Vrais Positifs  (TP) : **93** ✅ churners détectés

        *Chaque faux négatif = 845€ de revenu perdu en moyenne*
        """)

    with col2:
        st.success("""
        **Modèle retenu — Classification : Random Forest** 🌲
        - Meilleur ROC-AUC : **0.799**
        - Meilleur F1-Score : **0.361**
        - Feature importance native ✅

        **Modèle retenu — Régression : MLP** 🧠
        - Meilleure MAE : **138.85€**
        - Meilleur R² : **0.070**
        """)

# ============================================================
# PAGE 4 — PRÉDICTION CLIENT
# Appelle l'API FastAPI pour les prédictions
# ============================================================
elif page == "🔮 Prédiction client":

    st.title("🔮 Simulation — Prédiction pour un Client")
    st.markdown("Entrez les caractéristiques d'un client pour obtenir sa probabilité de churn et son revenu à risque.")
    st.info("💡 Les prédictions sont générées par l'API FastAPI — Classification : **Random Forest** | Régression : **MLP**")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Profil Client")
        age = st.slider("Âge", 18, 74, 35)
        gender = st.selectbox("Genre", ["Male", "Female"])
        country = st.selectbox("Pays", [
            "Bangladesh", "Canada", "Germany",
            "Australia", "India", "USA", "UK"
        ])
        customer_segment = st.selectbox(
            "Segment", ["Individual", "SME", "Enterprise"]
        )
        tenure_months = st.slider("Ancienneté (mois)", 1, 59, 12)
        signup_channel = st.selectbox("Canal", ["Web", "Mobile", "Referral"])
        contract_type = st.selectbox(
            "Contrat", ["Monthly", "Quarterly", "Yearly"]
        )

    with col2:
        st.subheader("📱 Utilisation")
        monthly_logins = st.slider("Connexions/mois", 0, 54, 15)
        weekly_active_days = st.slider("Jours actifs/semaine", 0, 7, 3)
        avg_session_time = st.slider("Durée session (min)", 1.0, 42.0, 15.0)
        features_used = st.slider("Features utilisées", 1, 15, 5)
        usage_growth_rate = st.slider("Croissance usage", -0.58, 0.54, 0.0)
        last_login_days_ago = st.slider("Dernière connexion (jours)", 0, 80, 5)

    with col3:
        st.subheader("💳 Facturation & Support")
        monthly_fee = st.slider("Abonnement mensuel (€)", 10, 100, 30)
        total_revenue = st.slider("Revenu total (€)", 10, 5900, 500)
        payment_method = st.selectbox(
            "Paiement", ["Card", "PayPal", "Bank Transfer"]
        )
        payment_failures = st.slider("Échecs paiement", 0, 5, 0)
        discount_applied = st.selectbox("Remise", ["No", "Yes"])
        price_increase_last_3m = st.selectbox(
            "Hausse prix récente", ["No", "Yes"]
        )
        support_tickets = st.slider("Tickets support", 0, 7, 1)
        avg_resolution_time = st.slider("Temps résolution (h)", 1.0, 62.0, 24.0)
        complaint_type = st.selectbox(
            "Type plainte",
            ["No_Complaint", "Technical", "Billing", "Service"]
        )
        csat_score = st.slider("CSAT Score", 1.0, 5.0, 3.5)
        escalations = st.slider("Escalades", 0, 4, 0)
        email_open_rate = st.slider("Taux ouverture email", 0.1, 0.9, 0.5)
        marketing_click_rate = st.slider(
            "Taux clic marketing", 0.01, 0.5, 0.25
        )
        nps_score = st.slider("NPS Score", -100, 100, 20)
        survey_response = st.selectbox(
            "Satisfaction enquête",
            ["Satisfied", "Neutral", "Unsatisfied"]
        )
        referral_count = st.slider("Références", 0, 7, 1)

    st.markdown("---")

    if st.button("🔮 Prédire via API", type="primary", use_container_width=True):

        payload = {
            "gender": gender, "age": age, "country": country,
            "customer_segment": customer_segment,
            "tenure_months": tenure_months,
            "signup_channel": signup_channel,
            "contract_type": contract_type,
            "monthly_logins": monthly_logins,
            "weekly_active_days": weekly_active_days,
            "avg_session_time": avg_session_time,
            "features_used": features_used,
            "usage_growth_rate": usage_growth_rate,
            "last_login_days_ago": last_login_days_ago,
            "monthly_fee": monthly_fee,
            "total_revenue": total_revenue,
            "payment_method": payment_method,
            "payment_failures": payment_failures,
            "discount_applied": discount_applied,
            "price_increase_last_3m": price_increase_last_3m,
            "support_tickets": support_tickets,
            "avg_resolution_time": avg_resolution_time,
            "complaint_type": complaint_type,
            "csat_score": csat_score,
            "escalations": escalations,
            "email_open_rate": email_open_rate,
            "marketing_click_rate": marketing_click_rate,
            "nps_score": nps_score,
            "survey_response": survey_response,
            "referral_count": referral_count
        }

        try:
            with st.spinner("⏳ Appel de l'API en cours..."):
                response = requests.post(
                    "http://localhost:8000/predict",
                    json=payload,
                    timeout=10
                )

            if response.status_code == 200:
                result = response.json()

                proba_churn     = result['classification']['proba_churn']
                churn_pred      = result['classification']['churn_prediction']
                risk_level      = result['classification']['risk_level']
                model_clf       = result['classification']['model_used']
                revenue_at_risk = result['regression']['revenue_at_risk']
                model_reg       = result['regression']['model_used']
                recommendation  = result['recommendation']

                st.markdown("### 📊 Résultats de la Prédiction")

                col1, col2, col3 = st.columns(3)

                with col1:
                    color = "🔴" if proba_churn > 0.5 else \
                            "🟡" if proba_churn > 0.3 else "🟢"
                    st.metric(
                        label=f"{color} Probabilité de Churn",
                        value=f"{proba_churn:.1%}",
                        help=f"Modèle : {model_clf}"
                    )
                with col2:
                    st.metric(
                        label="💰 Revenu à Risque",
                        value=f"{revenue_at_risk:,.2f} €",
                        help=f"Modèle : {model_reg}"
                    )
                with col3:
                    risk_emoji = "🔴" if risk_level == "élevé" else \
                                 "🟡" if risk_level == "modéré" else "🟢"
                    st.metric(
                        label="⚠️ Niveau de Risque",
                        value=f"{risk_emoji} {risk_level.upper()}"
                    )

                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"🤖 Classification : **{model_clf}**")
                with col2:
                    st.caption(f"🤖 Régression : **{model_reg}**")

                # Jauge de risque
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh(['Risque'], [proba_churn],
                        color='#e74c3c', height=0.5)
                ax.barh(['Risque'], [1 - proba_churn],
                        left=[proba_churn],
                        color='#2ecc71', height=0.5)
                ax.axvline(x=0.5, color='black',
                           linestyle='--', alpha=0.5,
                           label='Seuil 50%')
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probabilité de churn')
                ax.set_title(
                    f'Probabilité de churn : {proba_churn:.1%}',
                    fontweight='bold'
                )
                ax.legend()
                st.pyplot(fig)
                plt.close()

                st.markdown("---")
                st.subheader("💡 Recommandation")

                if risk_level == "élevé":
                    st.error(f"⚠️ {recommendation}")
                elif risk_level == "modéré":
                    st.warning(f"🟡 {recommendation}")
                else:
                    st.success(f"✅ {recommendation}")

            else:
                st.error(f"❌ Erreur API : {response.status_code} — {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("""
            ❌ **Impossible de contacter l'API !**

            Lance d'abord l'API FastAPI :
            ```
            cd api
            uvicorn main:app --reload --port 8000
            ```
            """)
        except requests.exceptions.Timeout:
            st.error("❌ Timeout — L'API met trop de temps à répondre")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

# ============================================================
# PAGE 5 — SHAP
# Graphiques générés par le notebook 07_comparaison_finale.ipynb
# Top 10 features RF issues du notebook 04 :
# 1. csat_score (0.1616)
# 2. tenure_months (0.1165)
# 3. monthly_logins (0.1004)
# 4. total_revenue (0.0921)
# 5. payment_failures (0.0825)
# 6. avg_session_time (0.0382)
# 7. avg_resolution_time (0.0365)
# 8. last_login_days_ago (0.0364)
# 9. nps_score (0.0322)
# 10. email_open_rate (0.0317)
# ============================================================
elif page == "🔍 Interprétabilité SHAP":

    st.title("🔍 Interprétabilité — SHAP Values")
    st.markdown("""
    SHAP explique **pourquoi** le modèle fait une prédiction.
    Appliqué sur le **Random Forest** (meilleur modèle, ROC-AUC = 0.799).
    """)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Importance Globale des Features")
        if os.path.exists('../results/shap_summary_bar.png'):
            st.image('../results/shap_summary_bar.png',
                     use_container_width=True)
        else:
            st.warning("Lance le notebook 07_comparaison_finale.ipynb d'abord")

    with col2:
        st.subheader("🌈 Impact des Features (Beeswarm)")
        if os.path.exists('../results/shap_beeswarm.png'):
            st.image('../results/shap_beeswarm.png',
                     use_container_width=True)
        else:
            st.warning("Lance le notebook 07_comparaison_finale.ipynb d'abord")

    st.markdown("---")
    st.subheader("🎯 Explication d'un Client à Risque")

    if os.path.exists('../results/shap_waterfall_client.png'):
        st.image('../results/shap_waterfall_client.png',
                 use_container_width=True)
        st.info("""
        **Comment lire ce graphique :**
        - 🔴 Barres rouges → augmentent le risque de churn
        - 🔵 Barres bleues → diminuent le risque de churn
        - La longueur = l'importance de la variable
        """)
    else:
        st.warning("Lance le notebook 07_comparaison_finale.ipynb d'abord")

    st.markdown("---")

    # Top 10 features issues du notebook 04 (Random Forest)
    st.subheader("🏆 Top 10 Features — Random Forest (notebook 04)")

    top_features = pd.DataFrame({
        'Feature'   : [
            'csat_score', 'tenure_months', 'monthly_logins',
            'total_revenue', 'payment_failures', 'avg_session_time',
            'avg_resolution_time', 'last_login_days_ago',
            'nps_score', 'email_open_rate'
        ],
        'Importance': [
            0.1616, 0.1165, 0.1004, 0.0921, 0.0825,
            0.0382, 0.0365, 0.0364, 0.0322, 0.0317
        ]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        top_features['Feature'][::-1],
        top_features['Importance'][::-1],
        color='#27ae60', alpha=0.8
    )
    ax.set_xlabel('Importance')
    ax.set_title(
        'Top 10 Features — Random Forest\n(Feature Importance native)',
        fontweight='bold'
    )
    for bar, val in zip(bars, top_features['Importance'][::-1]):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9
        )
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Insights — issus des vrais résultats SHAP et EDA
    col1, col2 = st.columns(2)
    with col1:
        st.error("""
        **Variables qui AUGMENTENT le risque** 🔴
        - `payment_failures` élevé (0.0825)
        - `last_login_days_ago` élevé (0.0364)
        - `support_tickets` élevé
        - `csat_score` bas (0.1616 — le plus important !)
        - `nps_score` bas (0.0322)
        """)
    with col2:
        st.success("""
        **Variables qui DIMINUENT le risque** 🟢
        - `tenure_months` élevé (0.1165)
        - `monthly_logins` élevé (0.1004)
        - `csat_score` élevé
        - `total_revenue` élevé (0.0921)
        - `avg_session_time` élevé (0.0382)
        """)