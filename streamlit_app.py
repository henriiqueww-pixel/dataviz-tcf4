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

st.title('🍟 analise de Risco de Obesidade')

st.info('Este aplicativo visa evidenciar as situações de risco analisadas de acordo com o banco de dados!')

def ordenar_opcoes(lista):

    """
    Ordena uma lista de strings ignorando acentos e maiúsculas
    """

    def normalizar(texto):
        if isinstance(texto, str):
            return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower()
        return str(texto)
    
    return sorted(lista, key=normalizar)

# Traduzir os nomes do SHAPE
def traduzir_nomes_features(lista_nomes_tecnicos):

    """
    Traduz os nomes técnicos do Pipeline para Português legível e profissional.
    """

    # Dicionário atualizado com o estilo preferido
    mapa_nomes = {
        # --- Numéricas ---
        'num__imc': 'Índice de Massa Corporal (IMC)',
        'num__idade': 'Idade',
        
        # --- Binárias ---
        'bin__genero': 'Gênero',
        'bin__b_historico_familiar': 'Histórico Familiar',
        'bin__b_fuma': 'Hábito de Fumar',
        'bin__b_come_alimentos_caloricos': 'Consumo de Calóricos',
        'bin__b_monitora_calorias': 'Monitoramento de Calorias',
        
        # --- Categorias: Comer entre Refeições ---
        'cat__freq_come_fora_refeicao_no': 'Comer entre refeições (Nunca)',
        'cat__freq_come_fora_refeicao_Sometimes': 'Comer entre refeições (Às vezes)',
        'cat__freq_come_fora_refeicao_Frequently': 'Comer entre refeições (Frequentemente)',
        'cat__freq_come_fora_refeicao_Always': 'Comer entre refeições (Sempre)',
        
        # --- Categorias: Atividade Física ---
        'cat__qtd_atv_fisicas_Sedentario': 'Sedentarismo',
        'cat__qtd_atv_fisicas_Baixa_frequencia': 'Baixa Atividade Física',
        'cat__qtd_atv_fisicas_Moderada_frequencia': 'Atividade Física Moderada',
        'cat__qtd_atv_fisicas_Alta_frequencia': 'Alta Atividade Física',
        
        # --- Categorias: Água ---
        'cat__qtd_agua_Baixo_consumo': 'Baixo consumo de água',
        'cat__qtd_agua_Consumo_adequado': 'Consumo de água (Adequado)',
        'cat__qtd_agua_Alto_consumo': 'Alto consumo de água',
        
        # --- Categorias: Transporte ---
        'cat__meio_de_transporte_Automobile': 'Uso de Carro',
        'cat__meio_de_transporte_Public_Transportation': 'Transporte Público',
        'cat__meio_de_transporte_Motorbike': 'Uso de Moto',
        'cat__meio_de_transporte_Bike': 'Uso de Bicicleta',
        'cat__meio_de_transporte_Walking': 'Caminhada',

        # --- Categorias: Refeições (Complementado no mesmo estilo) ---
        'cat__qtd_refeicao_Tres_refeicoes_principais_por_dia': '3 Refeições principais/dia',
        'cat__qtd_refeicao_Duas_refeicoes_principais_por_dia': '2 Refeições principais/dia',
        'cat__qtd_refeicao_Uma_refeicao_principal_por_dia': '1 Refeição principal/dia',
        'cat__qtd_refeicao_Quatro_ou_mais_refeicoes_principais_por_dia': '4+ Refeições principais/dia',
        
        # --- Categorias: Vegetais ---
        'cat__qtd_vegetais_Sempre': 'Consumo de Vegetais (Sempre)',
        'cat__qtd_vegetais_As_vezes': 'Consumo de Vegetais (Às vezes)',
        'cat__qtd_vegetais_Raramente': 'Consumo de Vegetais (Raramente)',
        
        # --- Categorias: Telas/Internet ---
        'cat__qtd_tmp_na_internet_Uso_baixo': 'Tempo em Telas (Baixo)',
        'cat__qtd_tmp_na_internet_Uso_moderado': 'Tempo em Telas (Moderado)',
        'cat__qtd_tmp_na_internet_Uso_intenso': 'Tempo em Telas (Intenso)',
        
        # --- Categorias: Álcool ---
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
            # Fallback de segurança: Se aparecer algo novo, limpa o nome técnico
            limpo = nome.replace('num__', '').replace('cat__', '').replace('bin__', '').replace('_', ' ').title()
            nomes_traduzidos.append(limpo)
            
    return nomes_traduzidos
        
# Salvar o modelo em cache
@st.cache_resource 

# Carregar o modelo
def load_model():

    """
    Carrega o modelo treinado (.joblib) localmente ou via GitHub
    """

    # Tentativa Local
    try:
        return joblib.load('risco_obesidade_random_forest.joblib')
    except FileNotFoundError:
        pass

    # Tentativa Remota (GitHub)
    url_modelo = "https://github.com/henriiqueww-pixel/dataviz-tcf4/raw/refs/heads/master/Modelos/risco_obesidade_random_forest.joblib"
    
    try:
        response = requests.get(url_modelo)
        if response.status_code == 200:
            return joblib.load(io.BytesIO(response.content))
    except Exception:
        pass
    
    return None

# Criar e cachear o SHAPE
@st.cache_resource

