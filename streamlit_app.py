import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ================================
# CONFIGURAÇÃO DA PÁGINA
# ================================

st.set_page_config(
    page_title="Risco Educacional",
    page_icon="🎓",
    layout="wide"
)

# ================================
# CARREGAR MODELO
# ================================

@st.cache_resource
def carregar_modelo():

    caminhos = [
        "modelo_passos_magicos.pkl",
        "modelo_passos_magicos.joblib",
        "./modelo_passos_magicos.pkl",
        "./modelo_passos_magicos.joblib"
    ]

    for caminho in caminhos:

        if os.path.exists(caminho):

            modelo = joblib.load(caminho)
            return modelo

    st.error("❌ Modelo não encontrado. Verifique se o arquivo modelo_passos_magicos.pkl está no repositório.")
    st.stop()


# ================================
# INPUT DO USUÁRIO
# ================================

def input_usuario():

    st.sidebar.header("Dados do Aluno")

    idade = st.sidebar.number_input("Idade", 10, 25, 15)

    nota_portugues = st.sidebar.slider(
        "Nota Português",
        0.0,
        10.0,
        6.0
    )

    nota_matematica = st.sidebar.slider(
        "Nota Matemática",
        0.0,
        10.0,
        6.0
    )

    frequencia = st.sidebar.slider(
        "Frequência (%)",
        0,
        100,
        75
    )

    horas_estudo = st.sidebar.slider(
        "Horas de estudo por semana",
        0,
        40,
        10
    )

    dados = {
        "idade": idade,
        "nota_portugues": nota_portugues,
        "nota_matematica": nota_matematica,
        "frequencia": frequencia,
        "horas_estudo": horas_estudo
    }

    df = pd.DataFrame([dados])

    return df


# ================================
# ALINHAR COLUNAS
# ================================

def alinhar_colunas(df, model):

    try:

        ordem = model.feature_names_in_

        for c in ordem:

            if c not in df.columns:
                df[c] = np.nan

        return df[ordem]

    except:

        return df


# ================================
# PREVISÃO
# ================================

def prever(model, df):

    prob = model.predict_proba(df)[0][1]

    if prob > 0.7:
        risco = "Alto"

    elif prob > 0.4:
        risco = "Moderado"

    else:
        risco = "Baixo"

    return prob, risco


# ================================
# EXPLICAÇÃO SHAP
# ================================

def explicar_modelo(model, df):

    try:

        import shap

        st.subheader("🔎 Explicação do Modelo")

        if hasattr(model, "named_steps"):

            steps = list(model.named_steps.values())

            modelo_final = steps[-1]

            try:
                transformador = steps[0]
                X_trans = transformador.transform(df)
            except:
                X_trans = df

        else:

            modelo_final = model
            X_trans = df

        explainer = shap.TreeExplainer(modelo_final)

        shap_values = explainer.shap_values(X_trans)

        if isinstance(shap_values, list):

            valores = shap_values[1][0]

        else:

            valores = shap_values[0]

        importancia = pd.DataFrame({
            "variavel": df.columns,
            "impacto": valores[:len(df.columns)]
        })

        importancia["impacto_abs"] = importancia["impacto"].abs()

        importancia = importancia.sort_values(
            "impacto_abs",
            ascending=False
        ).drop(columns="impacto_abs")

        st.dataframe(importancia)

        st.bar_chart(
            importancia.set_index("variavel")
        )

    except Exception:

        st.info("⚠️ Explicação SHAP não disponível para este modelo.")


# ================================
# APP PRINCIPAL
# ================================

def main():

    st.title("🎓 Plataforma de Risco Educacional")

    st.write(
        "Sistema de previsão de risco escolar usando Machine Learning."
    )

    model = carregar_modelo()

    df = input_usuario()

    df = alinhar_colunas(df, model)

    st.subheader("📋 Dados Informados")

    st.dataframe(df)

    if st.button("Realizar Previsão"):

        prob, risco = prever(model, df)

        st.subheader("📊 Resultado")

        st.metric(
            "Probabilidade de risco",
            f"{prob:.2%}"
        )

        st.metric(
            "Classificação",
            risco
        )

        explicar_modelo(model, df)


# ================================
# EXECUTAR APP
# ================================

if __name__ == "__main__":
    main()
