# Importar biblioteca completa - padrão
import io
import unicodedata

# Importar biblioteca completa - terceiro
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

st.set_page_config(page_title="Análise de Risco de Obesidade", layout="wide")

st.title('🍟 Análise de Risco de Obesidade')
st.info('Este aplicativo visa evidenciar as situações de risco analisadas de acordo com o banco de dados!')

def ordenar_opcoes(lista):
    def normalizar(texto):
        if isinstance(texto, str):
            return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower()
        return str(texto)
    return sorted(lista, key=normalizar)

def traduzir_nomes_features(lista_nomes_tecnicos):
    mapa_nomes = {
        'num__imc': 'Índice de Massa Corporal (IMC)',
        'num__idade': 'Idade',
        'bin__genero': 'Gênero',
        'bin__b_historico_familiar': 'Histórico Familiar',
        'bin__b_fuma': 'Hábito de Fumar',
        'bin__b_come_alimentos_caloricos': 'Consumo de Calóricos',
        'bin__b_monitora_calorias': 'Monitoramento de Calorias',
        'cat__freq_come_fora_refeicao_no': 'Comer entre refeições (Nunca)',
        'cat__freq_come_fora_refeicao_Sometimes': 'Comer entre refeições (Às vezes)',
        'cat__freq_come_fora_refeicao_Frequently': 'Comer entre refeições (Frequentemente)',
        'cat__freq_come_fora_refeicao_Always': 'Comer entre refeições (Sempre)',
        'cat__qtd_atv_fisicas_Sedentario': 'Sedentarismo',
        'cat__qtd_atv_fisicas_Baixa_frequencia': 'Baixa Atividade Física',
        'cat__qtd_atv_fisicas_Moderada_frequencia': 'Atividade Física Moderada',
        'cat__qtd_atv_fisicas_Alta_frequencia': 'Alta Atividade Física',
        'cat__qtd_agua_Baixo_consumo': 'Baixo consumo de água',
        'cat__qtd_agua_Consumo_adequado': 'Consumo de água (Adequado)',
        'cat__qtd_agua_Alto_consumo': 'Alto consumo de água',
        'cat__meio_de_transporte_Automobile': 'Uso de Carro',
        'cat__meio_de_transporte_Public_Transportation': 'Transporte Público',
        'cat__meio_de_transporte_Motorbike': 'Uso de Moto',
        'cat__meio_de_transporte_Bike': 'Uso de Bicicleta',
        'cat__meio_de_transporte_Walking': 'Caminhada'
    }

    return [mapa_nomes.get(n, n.replace('_', ' ').title()) for n in lista_nomes_tecnicos]

@st.cache_resource
def load_model():
    try:
        return joblib.load('risco_obesidade_random_forest.joblib')
    except FileNotFoundError:
        url = "https://github.com/henriiqueww-pixel/dataviz-tcf4/raw/refs/heads/master/Modelos/risco_obesidade_random_forest.joblib"
        response = requests.get(url)
        return joblib.load(io.BytesIO(response.content))

@st.cache_resource
def _get_shap_explainer(_classifier):
    return shap.TreeExplainer(_classifier)

def configurar_sidebar():
    with st.sidebar:
        st.header("📌 Sobre o Projeto")
        st.info("""
        Projeto desenvolvido para o **Tech Challenge – Fase 4**  
        🎓 FIAP + Alura
        """)

def gerar_explicacao_shap(model, input_df):
    pre = model.named_steps['preprocess']
    clf = model.named_steps['clf']

    Xt = pre.transform(input_df)
    nomes_raw = pre.get_feature_names_out()
    nomes_pt = traduzir_nomes_features(nomes_raw)

    df_map = pd.DataFrame({
        'Nome Técnico (Raw)': nomes_raw,
        'Nome Traduzido': nomes_pt,
        'Valor Inputado': Xt[0]
    })

    explainer = _get_shap_explainer(clf)
    shap_values = explainer(Xt)
    shap_values.feature_names = nomes_pt

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0, :, 1], show=False, max_display=10)

    return plt.gcf(), df_map

def get_user_input_features():
    st.header("1. Dados Pessoais")

    col1, col2 = st.columns(2)
    with col1:
        idade = st.number_input("Idade", 10, 100, 25)
        altura = st.number_input("Altura (m)", 1.0, 2.5, 1.70)

    with col2:
        genero_label = st.selectbox("Gênero", ["Masculino", "Feminino"])
        peso = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)

    genero = 1 if genero_label == "Feminino" else 0
    imc = int(np.ceil(peso / (altura ** 2)))

    st.header("2. Histórico")
    historico = st.radio("Histórico familiar?", ["Sim", "Não"], horizontal=True)
    fuma = st.radio("Você fuma?", ["Sim", "Não"], horizontal=True)

    st.header("3. Hábitos Alimentares")
    refeicao = st.selectbox("Refeições/dia", ["1", "2", "3", "4+"])
    vegetais = st.selectbox("Vegetais", ["Raramente", "Às vezes", "Sempre"])
    agua = st.selectbox("Água", ["< 1 Litro", "1-2 Litros", "> 2 Litros"])
    fora = st.selectbox("Come entre refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])

    st.header("4. Estilo de Vida")
    atv = st.selectbox("Atividade física", ["Sedentário", "Baixa", "Moderada", "Alta"])
    net = st.selectbox("Tempo em telas", ["Baixo (0-2h)", "Moderado (3-5h)", "Intenso (>5h)"])
    transporte = st.selectbox("Transporte", ["Carro", "Caminhada", "Bicicleta", "Moto", "Transporte Público"])

    data = {
        'idade': idade,
        'imc': imc,
        'genero': genero,
        'b_fuma': 1 if fuma == "Sim" else 0,
        'b_historico_familiar': 1 if historico == "Sim" else 0,
        'b_come_alimentos_caloricos': 0,
        'b_monitora_calorias': 0,
        'qtd_refeicao': refeicao,
        'qtd_vegetais': vegetais,
        'qtd_agua': agua,
        'freq_come_fora_refeicao': fora,
        'freq_alcool': 'no',
        'qtd_atv_fisicas': atv,
        'qtd_tmp_na_internet': net,
        'meio_de_transporte': transporte
    }

    return pd.DataFrame(data, index=[0])

def main():
    configurar_sidebar()
    model = load_model()

    input_df = get_user_input_features()

    validar_shap = st.checkbox("🔎 Mostrar debug técnico do SHAP")

    if st.button("🔍 Realizar Predição", use_container_width=True):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ ALTO RISCO ({prob*100:.1f}%)")
        else:
            st.success(f"✅ BAIXO RISCO ({prob*100:.1f}%)")

        fig, df_map = gerar_explicacao_shap(model, input_df)
        st.pyplot(fig)

        if validar_shap:
            st.dataframe(df_map, use_container_width=True)

if __name__ == "__main__":
    main()

