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
   # DADOS PESSOAIS
    st.header("1. Dados Pessoais")
    st.markdown("Inicie informando as características físicas básicas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        idade = st.number_input("Idade", min_value=10, max_value=100, value=25)
        altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70)
    
    with col2:
        genero_label = st.selectbox("Gênero", ordenar_opcoes(["Masculino", "Feminino"]))
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)

    # Cálculo de IMC e Gênero
    imc = int(np.ceil(peso / (altura ** 2)))
    genero = 1 if genero_label == "Feminino" else 0
    
    if imc < 18.5:
        tabela_imc = 'Abaixo do peso'

    elif imc >= 18.5 and imc <= 24.9:
        tabela_imc = 'Peso normal'

    elif imc >= 25.0 and imc <= 29.9:
        tabela_imc = 'Sobrepeso'

    elif imc >= 30.0 and imc <= 34.9:
        tabela_imc = 'Obesidade grau I'

    elif imc >= 35.0 and imc <= 39.9:
        tabela_imc = 'Obesidade grau II'

    else:
        tabela_imc = 'Obesidade grau III'

    st.info(f"ℹ️ **IMC Calculado:** {imc} kg/m² ({tabela_imc})")
    st.markdown("---")

    # HISTÓRICO E HÁBITOS
    st.header("2. Histórico e Monitoramento")
    
    col_h1, col_h2 = st.columns(2)
    
    with col_h1:
        historico = st.radio("Possui histórico familiar de sobrepeso?", ["Sim", "Não"], horizontal=True)
        fuma = st.radio("Você fuma?", ["Sim", "Não"], horizontal=True)
    
    with col_h2:
        caloricos = st.radio("Consome alimentos calóricos frequentemente?", ["Sim", "Não"], horizontal=True)
        monitora = st.radio("Costuma monitorar as calorias ingeridas?", ["Sim", "Não"], horizontal=True)

    b_historico_familiar = 1 if historico == "Sim" else 0
    b_fuma = 1 if fuma == "Sim" else 0
    b_come_alimentos_caloricos = 1 if caloricos == "Sim" else 0
    b_monitora_calorias = 1 if monitora == "Sim" else 0

    st.markdown("---")

    # HÁBITOS ALIMENTARES
    st.header("3. Hábitos Alimentares")

    mapa_refeicoes = {
        '1': 'Uma_refeicao_principal_por_dia',
        '2': 'Duas_refeicoes_principais_por_dia',
        '3': 'Tres_refeicoes_principais_por_dia',
        '4+': 'Quatro_ou_mais_refeicoes_principais_por_dia'
    }
    mapa_vegetais = {'Raramente': 'Raramente', 'Às vezes': 'As_vezes', 'Sempre': 'Sempre'}
    mapa_agua = {'< 1 Litro': 'Baixo_consumo', '1-2 Litros': 'Consumo_adequado', '> 2 Litros': 'Alto_consumo'}
    mapa_fora_hora = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
    mapa_alcool = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}

    col_alim1, col_alim2 = st.columns(2)

    with col_alim1:
        refeicao_key = st.selectbox(
            "Quantas refeições principais faz por dia?", 
            options=sorted(['1', '2', '3', '4+'])
        )
        veg_key = st.selectbox(
            "Frequência de consumo de vegetais?", 
            options=['Raramente', 'Às vezes', 'Sempre']
        )
        agua_key = st.selectbox(
            "Consumo diário de água?", 
            options=['< 1 Litro', '1-2 Litros', '> 2 Litros']
        )

    with col_alim2:
        fora_key = st.selectbox(
            "Costuma comer entre as refeições?", 
            options=list(mapa_fora_hora.keys())
        )
        alcool_key = st.selectbox(
            "Consome bebidas alcoólicas?", 
            options=list(mapa_alcool.keys())
        )

    qtd_refeicao = mapa_refeicoes[refeicao_key]
    qtd_vegetais = mapa_vegetais[veg_key]
    qtd_agua = mapa_agua[agua_key]
    freq_come_fora_refeicao = mapa_fora_hora[fora_key]
    freq_alcool = mapa_alcool[alcool_key]

    st.markdown("---")

    # ESTILO DE VIDA
    st.header("4. Estilo de Vida")

    mapa_atv = {
        'Sedentário': 'Sedentario', 
        'Baixa': 'Baixa_frequencia', 
        'Moderada': 'Moderada_frequencia', 
        'Alta': 'Alta_frequencia'
    }
    mapa_net = {
        'Baixo (0-2h)': 'Uso_baixo', 
        'Moderado (3-5h)': 'Uso_moderado', 
        'Intenso (>5h)': 'Uso_intenso'
    }
    mapa_transporte = {
        'Transporte Público': 'Public_Transportation', 
        'Caminhada': 'Walking', 
        'Carro': 'Automobile', 
        'Bicicleta': 'Bike', 
        'Moto': 'Motorbike'
    }

    col_estilo1, col_estilo2 = st.columns(2)

    with col_estilo1:
        atv_key = st.selectbox(
            "Frequência de atividade física?", 
            options=list(mapa_atv.keys())
        )
        net_key = st.selectbox(
            "Tempo diário em dispositivos eletrônicos?", 
            options=list(mapa_net.keys())
        )

    with col_estilo2:
        transporte_key = st.selectbox(
            "Meio de transporte principal?", 
            options=ordenar_opcoes(list(mapa_transporte.keys()))
        )

    qtd_atv_fisicas = mapa_atv[atv_key]
    qtd_tmp_na_internet = mapa_net[net_key]
    meio_de_transporte = mapa_transporte[transporte_key]

    data = {
        'idade': idade,
        'genero': genero,
        'qtd_refeicao': qtd_refeicao,
        'qtd_vegetais': qtd_vegetais,
        'qtd_agua': qtd_agua,
        'qtd_atv_fisicas': qtd_atv_fisicas,
        'qtd_tmp_na_internet': qtd_tmp_na_internet,
        'b_fuma': b_fuma,
        'b_come_alimentos_caloricos': b_come_alimentos_caloricos,
        'b_monitora_calorias': b_monitora_calorias,
        'b_historico_familiar': b_historico_familiar,
        'freq_come_fora_refeicao': freq_come_fora_refeicao,
        'freq_alcool': freq_alcool,
        'meio_de_transporte': meio_de_transporte,
        'imc': imc
    }
    
    return pd.DataFrame(data, index=[0])

def exibir_importancia_variaveis(model):

    """
    Extrai, formata e exibe as 3 variáveis mais importantes para o modelo.
    """

    # Acessar os passos do Pipeline
    # 'preprocess' é o nome do ColumnTransformer e 'clf' é o classificador
    preprocessor = model.named_steps['preprocess']
    classifier = model.named_steps['clf']

    # Obter os nomes das features transformadas (OneHot + Numéricas)
    feature_names = preprocessor.get_feature_names_out()
    
    # Obter os valores de importância
    importances = classifier.feature_importances_

    # Criar um DataFrame para organizar
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    df_imp = df_imp.sort_values('importance', ascending=False).head(3)

    # Dicionário para traduzir os nomes técnicos ("feature names") para Português legível
    dicionario_traducao = {
        'num__imc': 'Índice de Massa Corporal (IMC)',
        'num__idade': 'Idade',
        'bin__b_historico_familiar': 'Histórico Familiar',
        'bin__genero': 'Gênero',
        'bin__b_come_alimentos_caloricos': 'Consumo de Calóricos',
        'bin__b_fuma': 'Hábito de Fumar',
        'bin__b_monitora_calorias': 'Monitoramento de Calorias',
        'cat__freq_come_fora_refeicao_Frequently': 'Comer entre refeições (Frequentemente)',
        'cat__freq_come_fora_refeicao_Sometimes': 'Comer entre refeições (Às vezes)',
        'cat__freq_come_fora_refeicao_Always': 'Comer entre refeições (Sempre)',
        'cat__qtd_atv_fisicas_Sedentario': 'Sedentarismo',
        'cat__qtd_atv_fisicas_Baixa_frequencia': 'Baixa Atividade Física',
        'cat__qtd_atv_fisicas_Moderada_frequencia': 'Atividade Física Moderada',
        'cat__qtd_agua_Baixo_consumo': 'Baixo consumo de água',
        'cat__meio_de_transporte_Automobile': 'Uso de Carro',
        'cat__meio_de_transporte_Public_Transportation': 'Transporte Público'
    }

    # Função para limpar o nome
    def limpar_nome(nome_tecnico):
        if nome_tecnico in dicionario_traducao:
            return dicionario_traducao[nome_tecnico]
        
        nome_limpo = nome_tecnico.replace('num__', '').replace('cat__', '').replace('bin__', '')
        return nome_limpo.replace('_', ' ').title()

    # Aplicar a tradução
    df_imp['nome_exibicao'] = df_imp['feature'].apply(limpar_nome)

    # Exibição no Streamlit
    st.markdown("### 📊 Fatores de Maior Peso")
    st.markdown("As 3 principais variáveis que o modelo considerou para esta análise global:")

    for i, row in df_imp.iterrows():
        st.write(f"**{row['nome_exibicao']}**")
        st.progress(int(row['importance'] * 100))
        st.caption(f"Impacto no modelo: {row['importance']*100:.1f}%")

# Função princial
def main():
    # 1. Configura a Barra Lateral
    configurar_sidebar()

    # 2. Carrega o Modelo
    model = load_model()

    # 3. Corpo Principal
    st.title("🩺 Análise de Risco de Obesidade")
    st.markdown("""
    Preencha o formulário abaixo com os dados do paciente.
    O sistema utilizará Inteligência Artificial para calcular a probabilidade de risco de obesidade.
    """)
    st.markdown("---")

    # 4. Formulário
    input_df = get_user_input_features()

    # 5. Botão e Predição
    st.markdown("###")
    
    if st.button("🔍 Realizar Predição", type="primary", use_container_width=True):
        if model is not None:
            try:
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)

                st.markdown("---")
                st.header("Resultado da Análise")

                if prediction[0] == 1:
                    st.error("⚠️ **ALTO RISCO DE OBESIDADE IDENTIFICADO**")
                    st.metric(label="Probabilidade de Risco", value=f"{probability[0][1] * 100:.1f}%")
                    st.warning("👉 **Recomendação:** Sugere-se encaminhamento para orientação médica e nutricional especializada.")
                else:
                    st.success("✅ **BAIXO RISCO IMEDIATO**")
                    st.metric(label="Probabilidade de Risco", value=f"{probability[0][1] * 100:.1f}%")
                    st.info("👉 **Recomendação:** Continue mantendo hábitos saudáveis e acompanhamento regular.")
                
                # Exibição do SHAP
                st.markdown("---")
                st.header("Fatores de Influência (Explicabilidade)")
                st.write("Entenda quais fatores específicos deste paciente **aumentaram (Vermelho)** ou **diminuíram (Azul)** o risco.")
                
                with st.spinner("Calculando impactos detalhados..."):
                    fig_shap, df_map = gerar_explicacao_shap(model, input_df)
                    st.pyplot(fig_shap)
                    
                    st.markdown("""
                    **Legenda do Gráfico:**  
                    - **Eixo X:** Probabilidade de Risco.  
                    - **Barras Vermelhas:** Fatores que "empurram" o risco para cima.  
                    - **Barras Azuis:** Fatores que "seguram" o risco para baixo.  
                    """)

                # Validar SHAP
                if validar_shap.lower() == 's':

                    st.markdown("---")
                    st.header("🕵️‍♀️ Debug: Ver Mapeamento Técnico das Variáveis")
                    st.write("Verifique abaixo como cada variável técnica foi traduzida para o gráfico. Útil para encontrar duplicidades.")

                    with st.expander("Clique aqui para ver"):
                        st.dataframe(
                            df_map.sort_values(by='Nome Técnico (Raw)'), 
                            width='stretch',
                            hide_index=True
                        )

                # Exibição as principiais variaveis
                #st.markdown("---")
                #exibir_importancia_variaveis(model)
            
            except Exception as e:
                st.error(f"Ocorreu um erro técnico ao realizar a predição: {e}")
        else:
            st.error("⚠️ O modelo de Inteligência Artificial não foi carregado corretamente. Verifique os arquivos.")
            
if __name__ == "__main__":
    main()
