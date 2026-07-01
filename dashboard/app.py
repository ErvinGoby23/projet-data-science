# ============================================================
# DASHBOARD STREAMLIT — Rétention Client & Risque de Revenus
# Classification : Random Forest | Régression : XGBoost
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests
import os

st.set_page_config(
    page_title="Rétention Client",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #0F172A; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="metric-container"] { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; }
h1 { font-size: 24px !important; font-weight: 700 !important; color: #0F172A !important; }
h2 { font-size: 18px !important; font-weight: 600 !important; color: #1E293B !important; }
.stButton > button[kind="primary"] { background: #2563EB !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; height: 46px; }
hr { border-color: #E2E8F0 !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

PALETTE_MAIN   = '#2563EB'
PALETTE_DANGER = '#DC2626'
PALETTE_OK     = '#16A34A'
PALETTE_WARN   = '#D97706'

@st.cache_resource
def load_models():
    rf_clf       = joblib.load('../models/rf_classification.pkl')
    xgb_reg      = joblib.load('../models/xgb_regression.pkl')
    preprocessor = joblib.load('../models/preprocessor.pkl')
    return rf_clf, xgb_reg, preprocessor

@st.cache_data
def load_results():
    clf_files = {
        'Logistic Regression': '../results/lr_clf_results.csv',
        'Random Forest':       '../results/rf_clf_results.csv',
        'XGBoost':             '../results/xgb_clf_results.csv',
        'MLP':                 '../results/mlp_clf_results.csv',
    }
    reg_files = {
        'Ridge Regression': '../results/lr_reg_results.csv',
        'Random Forest':    '../results/rf_reg_results.csv',
        'XGBoost':          '../results/xgb_reg_results.csv',
        'MLP':              '../results/mlp_reg_results.csv',
    }

    clf_rows = []
    for name, path in clf_files.items():
        try:
            row = pd.read_csv(path).iloc[0]
            clf_rows.append({
                'Modèle'   : name,
                'Accuracy' : round(float(row['accuracy']), 4),
                'Precision': round(float(row['precision']), 4),
                'Recall'   : round(float(row['recall']), 4),
                'F1-Score' : round(float(row['f1']), 4),
                'ROC-AUC'  : round(float(row['roc_auc']), 4),
                'PR-AUC'   : round(float(row['pr_auc']), 4),
            })
        except Exception as e:
            st.warning(f"Erreur chargement {name} : {e}")

    reg_rows = []
    for name, path in reg_files.items():
        try:
            row = pd.read_csv(path).iloc[0]
            reg_rows.append({
                'Modèle'  : name,
                'MAE (€)' : round(float(row['mae']), 2),
                'RMSE (€)': round(float(row['rmse']), 2),
                'R²'      : round(float(row['r2']), 3),
            })
        except Exception as e:
            st.warning(f"Erreur chargement {name} : {e}")

    return pd.DataFrame(clf_rows), pd.DataFrame(reg_rows)

rf_clf, xgb_reg, preprocessor = load_models()

with st.sidebar:
    st.markdown("### Rétention Client")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🤖 Comparaison des modèles",
         "🔮 Prédiction client"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#64748B; line-height:1.8;'>
    <b style='color:#94A3B8'>Projet M1 Data Science</b><br>
    EFREI — RNCP40875 Bloc 2<br><br>
    <b style='color:#94A3B8'>Modèles retenus</b><br>
    Classification : Random Forest<br>
    Régression : XGBoost
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 1 — COMPARAISON DES MODÈLES
# ============================================================
if page == "🤖 Comparaison des modèles":
    st.title("Comparaison des 4 Modèles")
    st.markdown("<p style='color:#64748B; margin-top:-8px;'>Métriques calculées sur le jeu de test — 2 000 clients</p>", unsafe_allow_html=True)
    st.markdown("---")

    df_results, df_reg = load_results()

    st.subheader("Classification — Tableau comparatif")

    modeles_selectionnes = st.multiselect(
        "Filtrer les modèles affichés",
        options=df_results['Modèle'].tolist(),
        default=df_results['Modèle'].tolist()
    )
    df_results = df_results[df_results['Modèle'].isin(modeles_selectionnes)]

    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: #DCFCE7; color: #15803D; font-weight: 600' if v else '' for v in is_max]
    st.dataframe(
        df_results.style.apply(highlight_best, subset=['Recall'])
        .format({col: '{:.4f}' for col in ['Accuracy','Precision','Recall','F1-Score','ROC-AUC','PR-AUC']}),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    fig = go.Figure()
    metriques = ['Recall', 'ROC-AUC', 'F1-Score', 'Precision']
    colors    = ['#94A3B8', '#2563EB', '#7C3AED', '#0891B2']
    for i, row in df_results.iterrows():
        fig.add_trace(go.Bar(
            name=row['Modèle'], x=metriques,
            y=[row[m] for m in metriques],
            marker_color=colors[i % len(colors)]
        ))
    fig.update_layout(barmode='group', height=400,
                      title='Comparaison des métriques clés — Classification',
                      yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Régression — Revenue at Risk")
    def highlight_reg(s):
        is_best = s == s.min() if s.name in ['MAE (€)', 'RMSE (€)'] else s == s.max()
        return ['background-color: #DCFCE7; color: #15803D; font-weight: 600' if v else '' for v in is_best]
    st.dataframe(
        df_reg.style.apply(highlight_reg, subset=['MAE (€)','RMSE (€)','R²'])
        .format({'MAE (€)': '{:.2f}', 'RMSE (€)': '{:.2f}', 'R²': '{:.3f}'}),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#F0FDF4; border:1px solid #BBF7D0; border-radius:12px; padding:16px;'>
        <p style='font-weight:700; color:#15803D; margin:0 0 8px;'> Classification — Random Forest</p>
        <p style='color:#166534; font-size:13px; margin:0; line-height:1.8;'>
        ROC-AUC : <b>0.802</b> | Recall : <b>0.873</b><br>
        F1-Score : <b>0.377</b> | PR-AUC : <b>0.297</b><br>
        Rééquilibrage : <b>Undersampling</b></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px; padding:16px;'>
        <p style='font-weight:700; color:#1D4ED8; margin:0 0 8px;'>Régression — XGBoost</p>
        <p style='color:#1E40AF; font-size:13px; margin:0; line-height:1.8;'>
        MAE : <b>49.28 €</b> | RMSE : <b>130.96 €</b><br>
        R² : <b>0.981</b><br>
        Entraîné sur churners uniquement (<b>817 clients</b>)</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 2 — PRÉDICTION CLIENT
# ============================================================
elif page == "🔮 Prédiction client":
    st.title("Prédiction pour un Client")
    st.markdown("<p style='color:#64748B; margin-top:-8px;'>Simulation via l'API FastAPI — RF (classification) | XGBoost (régression)</p>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Bouton profil à risque élevé ---
    st.markdown("**Profil de démonstration**")
    use_high_risk = st.toggle("🔴 Charger un profil à risque élevé", value=False)

    # Valeurs par défaut selon le profil
    if use_high_risk:
        d = dict(
            age=52, gender="Male", country="Bangladesh",
            customer_segment="Individual", tenure_months=2,
            signup_channel="Web", contract_type="Monthly",
            monthly_logins=2, weekly_active_days=1,
            avg_session_time=3.0, features_used=1,
            usage_growth_rate=-0.50, last_login_days_ago=70,
            monthly_fee=90, payment_method="Card",
            payment_failures=5, discount_applied="No",
            price_increase_last_3m="Yes",
            support_tickets=7, avg_resolution_time=58.0,
            complaint_type="Billing", csat_score=1.0,
            escalations=4, email_open_rate=0.1,
            marketing_click_rate=0.01, nps_score=-80,
            survey_response="Unsatisfied", referral_count=0
        )
    else:
        d = dict(
            age=35, gender="Male", country="USA",
            customer_segment="SME", tenure_months=12,
            signup_channel="Web", contract_type="Quarterly",
            monthly_logins=15, weekly_active_days=3,
            avg_session_time=15.0, features_used=5,
            usage_growth_rate=0.0, last_login_days_ago=5,
            monthly_fee=30, payment_method="Card",
            payment_failures=0, discount_applied="No",
            price_increase_last_3m="No",
            support_tickets=1, avg_resolution_time=24.0,
            complaint_type="No_Complaint", csat_score=3.5,
            escalations=0, email_open_rate=0.5,
            marketing_click_rate=0.25, nps_score=20,
            survey_response="Satisfied", referral_count=1
        )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Profil")
        age              = st.slider("Âge", 18, 74, d['age'])
        gender           = st.selectbox("Genre", ["Male", "Female"], index=["Male","Female"].index(d['gender']))
        country          = st.selectbox("Pays", ["Bangladesh", "Canada", "Germany", "Australia", "India", "USA", "UK"], index=["Bangladesh","Canada","Germany","Australia","India","USA","UK"].index(d['country']))
        customer_segment = st.selectbox("Segment", ["Individual", "SME", "Enterprise"], index=["Individual","SME","Enterprise"].index(d['customer_segment']))
        tenure_months    = st.slider("Ancienneté (mois)", 1, 59, d['tenure_months'])
        signup_channel   = st.selectbox("Canal", ["Web", "Mobile", "Referral"], index=["Web","Mobile","Referral"].index(d['signup_channel']))
        contract_type    = st.selectbox("Contrat", ["Monthly", "Quarterly", "Yearly"], index=["Monthly","Quarterly","Yearly"].index(d['contract_type']))

    with col2:
        st.subheader("Utilisation")
        monthly_logins      = st.slider("Connexions / mois", 0, 54, d['monthly_logins'])
        weekly_active_days  = st.slider("Jours actifs / semaine", 0, 7, d['weekly_active_days'])
        avg_session_time    = st.slider("Durée session (min)", 1.0, 42.0, d['avg_session_time'])
        features_used       = st.slider("Features utilisées", 1, 15, d['features_used'])
        usage_growth_rate   = st.slider("Croissance usage", -0.58, 0.54, d['usage_growth_rate'])
        last_login_days_ago = st.slider("Dernière connexion (jours)", 0, 80, d['last_login_days_ago'])

    with col3:
        st.subheader("Facturation & Support")
        monthly_fee            = st.slider("Abonnement mensuel (€)", 10, 100, d['monthly_fee'])
        payment_method         = st.selectbox("Paiement", ["Card", "PayPal", "Bank Transfer"], index=["Card","PayPal","Bank Transfer"].index(d['payment_method']))
        payment_failures       = st.slider("Échecs paiement", 0, 5, d['payment_failures'])
        discount_applied       = st.selectbox("Remise", ["No", "Yes"], index=["No","Yes"].index(d['discount_applied']))
        price_increase_last_3m = st.selectbox("Hausse prix récente", ["No", "Yes"], index=["No","Yes"].index(d['price_increase_last_3m']))
        support_tickets        = st.slider("Tickets support", 0, 7, d['support_tickets'])
        avg_resolution_time    = st.slider("Temps résolution (h)", 1.0, 62.0, d['avg_resolution_time'])
        complaint_type         = st.selectbox("Type plainte", ["No_Complaint", "Technical", "Billing", "Service"], index=["No_Complaint","Technical","Billing","Service"].index(d['complaint_type']))
        csat_score             = st.slider("CSAT Score", 1.0, 5.0, d['csat_score'])
        escalations            = st.slider("Escalades", 0, 4, d['escalations'])
        email_open_rate        = st.slider("Taux ouverture email", 0.1, 0.9, d['email_open_rate'])
        marketing_click_rate   = st.slider("Taux clic marketing", 0.01, 0.5, d['marketing_click_rate'])
        nps_score              = st.slider("NPS Score", -100, 100, d['nps_score'])
        survey_response        = st.selectbox("Satisfaction enquête", ["Satisfied", "Neutral", "Unsatisfied"], index=["Satisfied","Neutral","Unsatisfied"].index(d['survey_response']))
        referral_count         = st.slider("Références", 0, 7, d['referral_count'])

    st.markdown("---")
    if st.button("🔮 Prédire via API FastAPI", type="primary", use_container_width=True):
        payload = {
            "gender": gender, "age": age, "country": country,
            "customer_segment": customer_segment, "tenure_months": tenure_months,
            "signup_channel": signup_channel, "contract_type": contract_type,
            "monthly_logins": monthly_logins, "weekly_active_days": weekly_active_days,
            "avg_session_time": avg_session_time, "features_used": features_used,
            "usage_growth_rate": usage_growth_rate, "last_login_days_ago": last_login_days_ago,
            "monthly_fee": monthly_fee,
            "payment_method": payment_method, "payment_failures": payment_failures,
            "discount_applied": discount_applied, "price_increase_last_3m": price_increase_last_3m,
            "support_tickets": support_tickets, "avg_resolution_time": avg_resolution_time,
            "complaint_type": complaint_type, "csat_score": csat_score,
            "escalations": escalations, "email_open_rate": email_open_rate,
            "marketing_click_rate": marketing_click_rate, "nps_score": nps_score,
            "survey_response": survey_response, "referral_count": referral_count
        }

        try:
            with st.spinner("Appel de l'API en cours..."):
                response = requests.post("http://localhost:8001/predict", json=payload, timeout=10)

            if response.status_code == 200:
                result          = response.json()
                proba_churn     = result['classification']['proba_churn']
                risk_level      = result['classification']['risk_level']
                revenue_at_risk = result['regression']['revenue_at_risk']
                recommendation  = result['recommendation']
                pct             = int(proba_churn * 100)

                if risk_level == "élevé":
                    color_risk, bg_risk, border_risk, icon_risk = PALETTE_DANGER, "#FEF2F2", "#FECACA", "🔴"
                elif risk_level == "modéré":
                    color_risk, bg_risk, border_risk, icon_risk = PALETTE_WARN, "#FFFBEB", "#FDE68A", "🟠"
                else:
                    color_risk, bg_risk, border_risk, icon_risk = PALETTE_OK, "#F0FDF4", "#BBF7D0", "🟢"

                st.markdown("---")
                st.subheader("Résultats de la Prédiction")
                c1, c2, c3 = st.columns(3)
                c1.metric("Probabilité de Churn", f"{pct}%")
                c2.metric("Revenu à Risque", f"{revenue_at_risk:,.2f} €")
                c3.metric("Niveau de Risque", risk_level.upper())

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pct,
                    title={'text': "Probabilité de churn (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color_risk},
                        'steps': [
                            {'range': [0, 30], 'color': '#DCFCE7'},
                            {'range': [30, 50], 'color': '#FEF9C3'},
                            {'range': [50, 100], 'color': '#FEE2E2'}
                        ],
                        'threshold': {'line': {'color': '#0F172A', 'width': 3},
                                      'thickness': 0.75, 'value': 50}
                    }
                ))
                fig.update_layout(height=280)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"""
                <div style='background:{bg_risk}; border:1px solid {border_risk}; border-radius:12px; padding:16px; margin-top:8px;'>
                    <p style='font-weight:700; color:{color_risk}; margin:0 0 6px; font-size:15px;'>{icon_risk} {risk_level.upper()} — Recommandation</p>
                    <p style='color:#374151; font-size:13px; margin:0; line-height:1.7;'>{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("Importance des Variables — Random Forest")
                BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                chemin        = os.path.join(BASE_DIR, 'data', 'processed', 'feature_names.csv')
                feature_names = pd.read_csv(chemin, header=None)[0].dropna().tolist()[1:51]
                importances   = rf_clf.feature_importances_
                feat_imp      = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feat_imp      = feat_imp.nlargest(10, 'Importance').sort_values('Importance')
                fig = go.Figure(go.Bar(
                    x=feat_imp['Importance'], y=feat_imp['Feature'],
                    orientation='h',
                    marker_color=[PALETTE_MAIN if v >= 0.10 else '#CBD5E1' for v in feat_imp['Importance']],
                    text=[f'{v:.4f}' for v in feat_imp['Importance']],
                    textposition='outside'
                ))
                fig.update_layout(height=380, xaxis_title='Importance',
                                  title='Top 10 features — Random Forest',
                                  xaxis_range=[0, feat_imp['Importance'].max() * 1.25])
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"Erreur API {response.status_code} — {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter l'API. Lancez : `uvicorn main:app --reload --port 8001`")
        except Exception as e:
            st.error(f"Erreur : {str(e)}")