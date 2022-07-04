# 1 Introdução

O projeto aborda um projeto de Machine Learning com o intuito de prever o sucesso de chamadas de telemarketing para a venda de depósitos bancários de longo prazo (disponível no link: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#). Foi abordado como fonte os dados de um banco português, com dados recolhidos de 2008 a 2013, incluindo dessa forma os efeitos da crise financeira vivida mundialmente a partir de 2009. A análise efetuada confirmou o modelo obtido como credível e valioso para os gestores de campanhas de telemarketing.

As campanhas de marketing foram baseadas em telefonemas e muitas vezes era necessário mais de um contato com o mesmo cliente para ter noção se o produto em análise (depósito bancário a prazo) seria ou não efetuado.

No banco de dados utilizado no projeto existe um conjunto de dados:
bank-full.csv: com todos os exemplos e 17 entradas, ordenadas por data (versão mais antiga deste conjunto de dados com menos entradas).

O objetivo da classificação é prever se o cliente irá subscrever um depósito a prazo (variável y).

Para o desenvolvimento deste estudo foram utilizadas algumas tecnologias, como Marchin Learn, K-means, Decision Tree, Pipeline e Weights & Biases.

# 2 Informação dos Atributos

A seguir serão apresentadas as informações dos atributos utilizados no projeto, detalhando, inclusive, a categorização de cada variável.

## 2.1 Variáveis de Entrada

### 2.1.1 Dados do cliente do banco:

* age: idade (numérico); 
* job: tipo de emprego (administrador, colarinho azul, empreendedor, empregada doméstica, gerenciamento, aposentado, autônomo, serviços, estudante, técnico, desempregado, desconhecido); 
* marital: estado civil (divorciado, casado, solteiro, desconhecido, nota: divorciado significa divorciado ou viúvo); 
* education: escolaridade (básico.4anos, básico.6 anos, básico.9 anos, ensino médio, analfabetos, curso profissional, grau universitário, desconhecido); 
* default: tem crédito inadimplente? (não, sim, desconhecido); 
* housing: tem crédito de habitação? (não, sim, desconhecido); 
* loan: tem empréstimo pessoal? (não, sim, desconhecido);

### 2.1.2 Dados relacionados com o último contato da campanha atual:

* contact: tipo de comunicação do contato (celular, telefone);
* month: último mês de contato do ano (jan, fev, mar, ..., nov, dec); 
* day of week: último dia de contato da semana (seg, ter, qua, qui, sex); 
* duration: duração do último contato, em segundos (numérico). Observação importante: esse atributo afeta muito o destino de saída (por exemplo, se duração = 0, então y = não). No entanto, a duração não é conhecida antes de uma chamada ser realizada. Além disso, após o término da chamada, y é obviamente conhecido.

### 2.1.3 Outros atributos:
* campaign: número de contatos realizados durante esta campanha e para este cliente (numérico, inclui último contato); 
* pdays: número de dias que se passaram após o último contato com o cliente de uma campanha anterior (numérico; 999 significa que o cliente não foi contatado anteriormente); 
* previous: número de contatos realizados antes desta campanha e para este cliente (numérico); 
* Poutcome: resultado da campanha de marketing anterior (fracasso, inexistente, sucesso);

###  2.1.4 Atributos do contexto social e econômico
* emp.var.rate: taxa de variação do emprego - indicador trimestral (numérico); 
* cons.price.idx: índice de preços ao consumidor - indicador mensal (numérico); 
* cons.conf.idx: índice de confiança do consumidor - indicador mensal (numérico); 
* euribor3m: taxa de 3 meses euribor - indicador diário (numérico); 
* nr.employed: número de funcionários - indicador trimestral (numérico).

## 2.2 Variáveis de Saída
* y - o cliente realizou um depósito a prazo? (sim, não).

# 3 Detalhes do <i> Workflow </i>
  
  Para a execução do projeto foram utilizados os passos apresentados no workflow criado e disponibilizado pelo Professor Ivanovitch Silva. Os passos desse pipeline foram conforme a imagem abaixo.
  
  ![Workflow](https://github.com/ivanovitchm/ppgeecmachinelearning/blob/main/images/workflow.png)

# 4 ETAPAS DO PROCESSO

## 4.1 Importanto o dataset

Para conhecer e baixar o arquivo dataset utilizado neste projeto acesse: (https://www.kaggle.com/datasets/krantiswalke/bankfullcsv)

## 4.2 Instalando bibliotecas 

~~~
!pip install pandas-profiling==3.1.0
!pip install wandb
~~~

## 4.3 Acessando o Wandb

Ants de realizar esta etapa será necessário criar um conta e adquirir a APIKEY do wandb. Acesse o link (https://wandb.ai) para registro. 
~~~
# Login to Weights & Biases
!wandb login --relogin
~~~

## 4.4 Importanto as bibliotecas 
~~~
%%file test_data.py
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas_profiling import ProfileReport
import tempfile
import os
import pytest
~~~
  
## 4.5 Importando o dataset
~~~
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
     
income = pd.read_csv("bank-full.csv", delimiter=';')
income.head()
~~~

## 4.6 Raw Data
Criar o arquivo .csv   
~~~
income.to_csv("raw_data.csv",index=False)
~~~
Enviando para o wandb como artefato.
~~~
!wandb artifact put \
      --name decision_tree/raw_data.csv \
      --type raw_data \
      --description "The raw data from bank marketing" raw_data.csv
~~~

## 4.7 EDA
Inicialmente vamos chamar (iniciar) o projeto decision_tree salvo no wandb.

~~~
run = wandb.init(project="decision_tree", save_code=True)
~~~

Fazer o download da última versão do artefato
~~~
# donwload the latest version of artifact raw_data.csv
artifact = run.use_artifact("decision_tree/raw_data.csv:latest")
~~~
Criar o dataframe
~~~
# create a dataframe from the artifact
df = pd.read_csv(artifact.file())
~~~

Visualização dos dados no Pandas Profiling
~~~
ProfileReport(df, title="Pandas Profiling Report", explorative=True)
~~~

## 4.8 Pré-processamento
Importar o artefato
~~~
input_artifact="decision_tree/raw_data.csv:latest"
artifact_name="preprocessed_data.csv"
artifact_type="clean_data"
artifact_description="Data after preprocessing"
~~~
Criar um novo job_type
~~~
run = wandb.init(project="decision_tree", job_type="process_data")
~~~
download da última versão do artefato e cria um DF
~~~
artifact = run.use_artifact(input_artifact)
df = pd.read_csv(artifact.file())
~~~
Cria um novo artefato e realiza as configurações básicas
~~~
artifact = wandb.Artifact(name=artifact_name,
                          type=artifact_type,
                          description=artifact_description)
artifact.add_file(artifact_name)
~~~
Faz o upload do artefato e finaliza
~~~
run.log_artifact(artifact)
run.finish()
~~~

## 4.9 Data_check
Inicia novamente o projeto
~~~
run = wandb.init(project="decision_tree", job_type="data_checks")
~~~
Define algumas colunas de teste
~~~
@pytest.fixture(scope="session")
def data():

    local_path = run.use_artifact("decision_tree/preprocessed_data.csv:latest").file()
    df = pd.read_csv(local_path)

    return df

def test_data_length(data):
    """
    We test that we have enough data to continue
    """
    assert len(data) > 1000


def test_number_of_columns(data):
    """
    We test that we have enough data to continue
    """
    assert data.shape[1] == 17

def test_column_presence_and_type(data):

    required_columns = {
        "age": pd.api.types.is_int64_dtype,
        "job": pd.api.types.is_object_dtype,
        "marital": pd.api.types.is_object_dtype,
        "education": pd.api.types.is_object_dtype,
        "default": pd.api.types.is_object_dtype,
        "balance": pd.api.types.is_int64_dtype,
        "housing": pd.api.types.is_object_dtype,
        "loan": pd.api.types.is_object_dtype,
        "contact": pd.api.types.is_object_dtype,
        "day": pd.api.types.is_int64_dtype,
        "month": pd.api.types.is_object_dtype,
        "duration": pd.api.types.is_int64_dtype,
        "campaign": pd.api.types.is_int64_dtype,
        "pdays": pd.api.types.is_int64_dtype,
        "previous": pd.api.types.is_int64_dtype,
        "poutcome": pd.api.types.is_object_dtype,
        "y": pd.api.types.is_object_dtype
    }

    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"

def test_column_ranges(data):

    ranges = {
        "age": (18, 95),
        "balance": (-8019, 102127),
        "day": (1, 31),
        "duration": (0, 4918),
        "campaign": (1, 63),
        "pdays": (-1, 871),
        "previous": (0, 275)
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={data[col_name].min()} and max={data[col_name].max()}"
        )
~~~
Tendo o código de teste faz a execução pelo terminal com o comando:
~~~
!pytest . -vv
~~~




