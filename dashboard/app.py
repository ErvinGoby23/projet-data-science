# ============================================================
# DASHBOARD STREAMLIT — Retention Client & Risque de Revenus
# Donnees issues des notebooks de modelisation
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
    page_title="Retention Client — Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CHARGEMENT DES MODELES ET DONNEES
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
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choisir une page",
    [
        "Vue d'ensemble",
        "Analyse des donnees",
        "Comparaison des modeles",
        "Prediction client",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Projet Data Science M1**")
st.sidebar.markdown("Retention Client & Risque de Revenus")
st.sidebar.markdown("Dataset : 10 000 clients SaaS")

# ============================================================
# PAGE 1 — VUE D'ENSEMBLE
# ============================================================
if page == "Vue d'ensemble":

    st.title("Systeme Intelligent de Retention Client")
    st.markdown("### Tableau de bord decisionnel — Vue d'ensemble")
    st.markdown("---")

    total_clients  = len(df)
    churners       = df['churn'].sum()
    taux_churn     = df['churn'].mean() * 100
    revenu_risque  = df[df['churn']==1]['total_revenue'].sum()
    revenu_moyen   = df['total_revenue'].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Clients",
            value=f"{total_clients:,}"
        )
    with col2:
        st.metric(
            label="Clients a Risque",
            value=f"{churners:,}",
            delta=f"{taux_churn:.1f}% du total",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            label="Revenu a Risque",
            value=f"{revenu_risque:,.0f} EUR",
            delta_color="inverse"
        )
    with col4:
        st.metric(
            label="Revenu Moyen/Client",
            value=f"{revenu_moyen:,.0f} EUR"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
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
        ax.set_title('Taux de churn par segment\n(SME = segment le plus a risque)')
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

   

# ============================================================
# PAGE 2 — ANALYSE DES DONNEES
# ============================================================
elif page == "Analyse des donnees":

    st.title("Analyse Exploratoire des Donnees")
    st.markdown("Resultats issus de l'EDA realisee sur le dataset de 10 000 clients.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn par Pays")
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_country = df.groupby('country')['churn'].mean() * 100
        churn_country.sort_values().plot(kind='barh', ax=ax, color='#3498db')
        ax.axvline(
            x=df['churn'].mean()*100, color='red',
            linestyle='--', label='Moyenne (10.2%)'
        )
        ax.set_xlabel('Taux de churn (%)')
        ax.set_title('Australia = pays le plus a risque (11.5%)')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
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


# ============================================================
# PAGE 3 — COMPARAISON DES MODELES
# ============================================================
elif page == "Comparaison des modeles":

    st.title("Comparaison des 4 Modeles")
    st.markdown("Metriques calculees sur le jeu de test (2000 clients, 10.2% churners).")
    st.markdown("---")

    results_data = {
    'Modele'   : ['Logistic Regression', 'Random Forest', 'XGBoost', 'MLP (Deep Learning)'],
    'Accuracy' : [0.6550, 0.7145, 0.7125, 0.6445],
    'Precision': [0.1743, 0.2434, 0.2300, 0.1353],
    'Recall'   : [0.6373, 0.8529, 0.7745, 0.4608],
    'F1-Score' : [0.2737, 0.3787, 0.3547, 0.2091],
    'ROC-AUC'  : [0.7094, 0.8103, 0.7806, 0.5596]
}
    df_results = pd.DataFrame(results_data)

    st.subheader("Tableau Comparatif — Classification")
    st.dataframe(
        df_results.style.highlight_max(
            subset=['Accuracy', 'Precision', 'Recall',
                    'F1-Score', 'ROC-AUC'],
            color='#d4edda'
        ),
        use_container_width=True
    )

    st.markdown("---")

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
    plt.suptitle('Comparaison des 4 modeles', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    st.subheader("Tableau Comparatif — Regression (Revenue at Risk)")
    reg_data = {
        'Modele'  : ['Ridge Regression', 'Random Forest',
                     'XGBoost', 'MLP (Deep Learning)'],
        'MAE (EUR)' : [156.11, 145.39, 158.61, 136.89],
        'RMSE (EUR)': [375.15, 373.57, 386.03, 372.30],
        'R2'        : [0.0533, 0.0612, -0.0024, 0.0676]
    }
    df_reg = pd.DataFrame(reg_data)
    st.dataframe(
        df_reg.style.highlight_min(
            subset=['MAE (EUR)', 'RMSE (EUR)'], color='#d4edda'
        ).highlight_max(
            subset=['R2'], color='#d4edda'
        ),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Analyse des Erreurs — Random Forest (modele final)")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Matrice de confusion — Random Forest**
        - Vrais Negatifs  (TN) : 1578 [OK]
        - Faux Positifs   (FP) : 218
        - Faux Negatifs   (FN) : 111 [ATTENTION : churners manques]
        - Vrais Positifs  (TP) : 93 [OK : churners detentes]
        """)

    with col2:
        st.success("""
        **Modele retenu — Classification : Random Forest**
        - Meilleur ROC-AUC : 0.799
        - Meilleur F1-Score : 0.361
        - Feature importance native OK

        **Modele retenu — Regression : MLP**
        - Meilleure MAE : 138.85 EUR
        - Meilleur R2 : 0.070
        """)

# ============================================================
# PAGE 4 — PREDICTION CLIENT
# ============================================================
elif page == "Prediction client":

    st.title("Simulation — Prediction pour un Client")
    st.markdown("Entrez les caracteristiques d'un client pour obtenir sa probabilite de churn et son revenu a risque.")
    st.info("Les predictions sont generees par l'API FastAPI — Classification : Random Forest | Regression : MLP")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Profil Client")
        age = st.slider("Age", 18, 74, 35)
        gender = st.selectbox("Genre", ["Male", "Female"])
        country = st.selectbox("Pays", [
            "Bangladesh", "Canada", "Germany",
            "Australia", "India", "USA", "UK"
        ])
        customer_segment = st.selectbox(
            "Segment", ["Individual", "SME", "Enterprise"]
        )
        tenure_months = st.slider("Anciennete (mois)", 1, 59, 12)
        signup_channel = st.selectbox("Canal", ["Web", "Mobile", "Referral"])
        contract_type = st.selectbox(
            "Contrat", ["Monthly", "Quarterly", "Yearly"]
        )

    with col2:
        st.subheader("Utilisation")
        monthly_logins = st.slider("Connexions/mois", 0, 54, 15)
        weekly_active_days = st.slider("Jours actifs/semaine", 0, 7, 3)
        avg_session_time = st.slider("Duree session (min)", 1.0, 42.0, 15.0)
        features_used = st.slider("Features utilisees", 1, 15, 5)
        usage_growth_rate = st.slider("Croissance usage", -0.58, 0.54, 0.0)
        last_login_days_ago = st.slider("Derniere connexion (jours)", 0, 80, 5)

    with col3:
        st.subheader("Facturation & Support")
        monthly_fee = st.slider("Abonnement mensuel (EUR)", 10, 100, 30)
        total_revenue = st.slider("Revenu total (EUR)", 10, 5900, 500)
        payment_method = st.selectbox(
            "Paiement", ["Card", "PayPal", "Bank Transfer"]
        )
        payment_failures = st.slider("Echecs paiement", 0, 5, 0)
        discount_applied = st.selectbox("Remise", ["No", "Yes"])
        price_increase_last_3m = st.selectbox(
            "Hausse prix recente", ["No", "Yes"]
        )
        support_tickets = st.slider("Tickets support", 0, 7, 1)
        avg_resolution_time = st.slider("Temps resolution (h)", 1.0, 62.0, 24.0)
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
            "Satisfaction enquete",
            ["Satisfied", "Neutral", "Unsatisfied"]
        )
        referral_count = st.slider("References", 0, 7, 1)

    st.markdown("---")

    if st.button("PREDIRE via API", type="primary", use_container_width=True):

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
            with st.spinner("Appel de l'API en cours..."):
                response = requests.post(
                    "http://localhost:8000/predict",
                    json=payload,
                    timeout=10
                )

            if response.status_code == 200:
                result = response.json()

                proba_churn     = result['classification']['proba_churn']
                risk_level      = result['classification']['risk_level']
                revenue_at_risk = result['regression']['revenue_at_risk']
                recommendation  = result['recommendation']

                st.markdown("### Resultats de la Prediction")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Probabilite de Churn",
                        value=f"{proba_churn:.1%}"
                    )
                with col2:
                    st.metric(
                        label="Revenu a Risque",
                        value=f"{revenue_at_risk:,.2f} EUR"
                    )
                with col3:
                    st.metric(
                        label="Niveau de Risque",
                        value=f"{risk_level.upper()}"
                    )

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
                ax.set_xlabel('Probabilite de churn')
                ax.set_title(
                    f'Probabilite de churn : {proba_churn:.1%}',
                    fontweight='bold'
                )
                st.pyplot(fig)
                plt.close()

                st.markdown("---")
                st.subheader("Recommandation")

                if risk_level == "eleve":
                    st.error(f"ATTENTION : {recommendation}")
                elif risk_level == "modere":
                    st.warning(f"ORANGE : {recommendation}")
                else:
                    st.success(f"OK : {recommendation}")

            else:
                st.error(f"ERREUR API : {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("ERREUR : Impossible de contacter l'API")
        except Exception as e:
            st.error(f"ERREUR : {str(e)}")

# ============================================================
# PAGE 5 — SHAP
# ============================================================
elif page == "Interpretabilite SHAP":

    st.title("Interpretabilite — SHAP Values")
    st.markdown("""
    SHAP explique **pourquoi** le modele fait une prediction.
    Applique sur le **Random Forest** (meilleur modele, ROC-AUC = 0.799).
    """)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Importance Globale des Features")
        if os.path.exists('../results/shap_summary_bar.png'):
            st.image('../results/shap_summary_bar.png',
                     use_container_width=True)

    with col2:
        st.subheader("Impact des Features (Beeswarm)")
        if os.path.exists('../results/shap_beeswarm.png'):
            st.image('../results/shap_beeswarm.png',
                     use_container_width=True)

    st.markdown("---")

    st.subheader("Top 10 Features — Random Forest")

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