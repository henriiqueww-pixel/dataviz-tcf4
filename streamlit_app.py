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
    """Ordena uma lista de strings ignorando acentos e maiúsculas"""
    def normalizar(texto):
        if isinstance(texto, str):
            return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower()
        return str(texto)
    return sorted(lista, key=normalizar)

def traduzir_nomes_features(lista_nomes_tecnicos):
    """Traduz os nomes técnicos do Pipeline para Português legível."""
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
        'cat__meio_de_transporte_Walking': 'Caminhada',
        'cat__qtd_refeicao_Tres_refeicoes_principais_por_dia': '3 Refeições principais/dia',
        'cat__qtd_refeicao_Duas_refeicoes_principais_por_dia': '2 Refeições principais/dia',
        'cat__qtd_refeicao_Uma_refeicao_principal_por_dia': '1 Refeição principal/dia',
        'cat__qtd_refeicao_Quatro_ou_mais_refeicoes_principais_por_dia': '4+ Refeições principais/dia',
        'cat__qtd_vegetais_Sempre': 'Consumo de Vegetais (Sempre)',
        'cat__qtd_vegetais_As_vezes': 'Consumo de Vegetais (Às vezes)',
        'cat__qtd_vegetais_Raramente': 'Consumo de Vegetais (Raramente)',
        'cat__qtd_tmp_na_internet_Uso_baixo': 'Tempo em Telas (Baixo)',
        'cat__qtd_tmp_na_internet_Uso_moderado': 'Tempo em Telas (Moderado)',
        'cat__qtd_tmp_na_internet_Uso_intenso': 'Tempo em Telas (Intenso)',
        'cat__freq_alcool_no': 'Consumo de Álcool (Não)',
        'cat__freq_alcool_Sometimes': 'Consumo de Álcool (Às vezes)',
        'cat__freq_alcool_Frequently': 'Consumo de Álcool (Frequentemente)',
        'cat__freq_alcool_Always': 'Consumo de Álcool (Sempre)'
    }
    
    nomes_traduzidos = []
    for nome in lista_nomes_tecnicos:
        if nome in mapa_nomes:
            nomes_traduzidos.append(mapa_nomes[nome])
        else:
            limpo = nome.replace('num__', '').replace('cat__', '').replace('bin__', '').replace('_', ' ').title()
            nomes_traduzidos.append(limpo)
    return nomes_traduzidos

@st.cache_resource
def load_model():
    """Carrega o modelo treinado localmente ou via GitHub"""
    try:
        return joblib.load('risco_obesidade_random_forest.joblib')
    except FileNotFoundError:
        url_modelo = "https://github.com/henriiqueww-pixel/dataviz-tcf4/raw/refs/heads/master/Modelos/risco_obesidade_random_forest.joblib"
        try:
            response = requests.get(url_modelo)
            if response.status_code == 200:
                return joblib.load(io.BytesIO(response.content))
        except Exception:
            return None
    return None

@st.cache_resource
def _get_shap_explainer(_classifier):
    """Cria e cacheia o explicador do SHAP."""
    return shap.TreeExplainer(_classifier)

def configurar_sidebar():
    with st.sidebar:
        st.header("📌 Sobre o Projeto")
        st.info("""
            Este aplicativo foi desenvolvido para o **Tech Challenge** da **Fase 4**.
            🎓 **Curso:** Pós-Graduação em Data Analytics  
            🏫 **Instituição:** FIAP + Alura
            """)

def gerar_explicacao_shap(model, input_df):
    preprocessor = model.named_steps['preprocess']
    classifier = model.named_steps['clf']
    input_transformed = preprocessor.transform(input_df)
    feature_names_raw = preprocessor.get_feature_names_out()
    feature_names_pt = traduzir_nomes_features(feature_names_raw)

    df_mapeamento = pd.DataFrame({
        'Nome Técnico (Raw)': feature_names_raw,
        'Nome Traduzido': feature_names_pt,  
        'Valor Inputado': input_transformed[0]
    })

    explainer = _get_shap_explainer(classifier)
    shap_values = explainer(input_transformed)
    shap_values.feature_names = feature_names_pt

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0, :, 1], show=False, max_display=10)
    
    return plt.gcf(), df_mapeamento

def get_user_input_features():
    st.header("1. Dados Pessoais")
    col1, col2 = st.columns(2)
    with col1:
        idade = st.number_input("Idade", min_value=10, max_value=100, value=25)
        altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70)
    with col2:
        genero_label = st.selectbox("Gênero", ordenar_opcoes(["Masculino", "Feminino"]))
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)

    imc = int(np.ceil(peso / (altura ** 2)))
    genero = 1 if genero_label == "Feminino" else 0
    st.info(f"ℹ️ **IMC Calculado:** {imc} kg/m²")

    st.header("2. Histórico e Monitoramento")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        historico = st.radio("Possui histórico familiar de sobrepeso?", ["Sim", "Não"], horizontal=True)
        fuma = st.radio("Você fuma?", ["Sim", "Não"], horizontal=True)
    with col_h2:
        caloricos = st.radio("Consome alimentos calóricos frequentemente?", ["Sim", "Não"], horizontal=True)
        monitora = st.radio("Costuma monitorar as calorias ingeridas?", ["Sim", "Não"], horizontal=True)

    st.header("3. Hábitos Alimentares")
    mapa_refeicoes = {'1': 'Uma_refeicao_principal_por_dia', '2': 'Duas_refeicoes_principais_por_dia', '3': 'Tres_refeicoes_principais_por_dia', '4+': 'Quatro_ou_mais_refeicoes_principais_por_dia'}
    col_alim1, col_alim2 = st.columns(2)
    with col_alim1:
        refeicao_key = st.selectbox("Refeições principais/dia?", options=['1', '2', '3', '4+'], index=2)
        veg_key = st.selectbox("Frequência de vegetais?", options=['Raramente', 'Às vezes', 'Sempre'], index=1)
    with col_alim2:
        agua_key = st.selectbox("Consumo de água?", options=['< 1 Litro', '1-2 Litros', '> 2 Litros'], index=1)
        fora_key = st.selectbox("Come entre as refeições?", options=['Não', 'Às vezes', 'Frequentemente', 'Sempre'], index=1)

    data = {
        'idade': idade, 'genero': genero, 'qtd_refeicao': mapa_refeicoes[refeicao_key],
        'qtd_vegetais': veg_key, 'qtd_agua': agua_key, 'qtd_atv_fisicas': 'Sedentario',
        'qtd_tmp_na_internet': 'Uso_moderado', 'b_fuma': 1 if fuma == "Sim" else 0,
        'b_come_alimentos_caloricos': 1 if caloricos == "Sim" else 0, 'b_monitora_calorias': 1 if monitora == "Sim" else 0,
        'b_historico_familiar': 1 if historico == "Sim" else 0, 'freq_come_fora_refeicao': 'Sometimes',
        'freq_alcool': 'no', 'meio_de_transporte': 'Public_Transportation', 'imc': imc
    }
    return pd.DataFrame(data, index=[0])

def main():
    configurar_sidebar()
    model = load_model()

    input_df = get_user_input_features()

    if st.button("🔍 Realizar Predição", type="primary", use_container_width=True):
        if model is not None:
            try:
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)

                st.markdown("---")
                if prediction[0] == 1:
                    st.error(f"⚠️ **ALTO RISCO DE OBESIDADE** ({probability[0][1] * 100:.1f}%)")
                else:
                    st.success(f"✅ **BAIXO RISCO IMEDIATO** ({probability[0][1] * 100:.1f}%)")
                
                with st.spinner("Gerando explicação..."):
                    fig_shap, df_map = gerar_explicacao_shap(model, input_df)
                    st.pyplot(fig_shap)

                if st.checkbox("Exibir Debug de Mapeamento"):
                    st.dataframe(df_map)

            except Exception as e:
                st.error(f"Erro na predição: {e}")
        else:
            st.error("Modelo não carregado.")

if __name__ == "__main__":
    main()
