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
import logging
import tempfile
from sklearn.model_selection import train_test_split
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

## 4.10 Data_segregation
Define algumas variáveis globais
Razão utilizada para dividir o treinamento e teste
~~~
test_size = 0.30
~~~

Seed (semente utilizada para reprodução dos mesmos valores)
~~~
seed = 41
~~~

Coluna de referência para estratificar os dados
~~~
stratify = "y"
~~~

Nome do artefato
~~~
artifact_input_name = "decision_tree/preprocessed_data.csv:latest"
~~~

Tipo do artefato
~~~
artifact_type = "segregated_data"
~~~

## configure logging
~~~
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')
~~~
Referência para um objeto logging
~~~
logger = logging.getLogger()
~~~

Ininica o projeto no wandb
~~~
run = wandb.init(project="decision_tree", job_type="split_data")

logger.info("Downloading and reading artifact")
artifact = run.use_artifact(artifact_input_name)
artifact_path = artifact.file()
df = pd.read_csv(artifact_path)
~~~

Split firstly in train/test, then we further divide the dataset to train and validation
~~~
logger.info("Splitting data into train and test")
splits = {}

splits["train"], splits["test"] = train_test_split(df,
                                                   test_size=test_size,
                                                   random_state=seed,
                                                   stratify=df[stratify])
~~~

Save the artifacts. We use a temporary directory so we do not leave any trace behind
~~~
with tempfile.TemporaryDirectory() as tmp_dir:

    for split, df in splits.items():

        # Make the artifact name from the name of the split plus the provided root
        artifact_name = f"{split}.csv"

        # Get the path on disk within the temp directory
        temp_path = os.path.join(tmp_dir, artifact_name)

        logger.info(f"Uploading the {split} dataset to {artifact_name}")

        # Save then upload to W&B
        df.to_csv(temp_path,index=False)

        artifact = wandb.Artifact(name=artifact_name,
                                  type=artifact_type,
                                  description=f"{split} split of dataset {artifact_input_name}",
        )
        artifact.add_file(temp_path)

        logger.info("Logging artifact")
        run.log_artifact(artifact)

        # This waits for the artifact to be uploaded to W&B. If you
        # do not add this, the temp directory might be removed before
        # W&B had a chance to upload the datasets, and the upload
        # might fail
        artifact.wait()
~~~

Finalizar o run para atualizar no wandb
~~~
run.finish()
~~~


## 4.11 Train

Importe das principais bibliotecas que serão utilizadas nesta etapa

~~~
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
~~~

Variaveis globais
~~~
# ratio used to split train and validation data
val_size = 0.30

# seed used to reproduce purposes
seed = 41

# reference (column) to stratify the data
stratify = "y"

# name of the input artifact
artifact_input_name = "decision_tree/train.csv:latest"

# type of the artifact
artifact_type = "Train"
~~~

Etapa de <b>logging</b>
~~~
# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

# initiate the wandb project
run = wandb.init(project="decision_tree",job_type="train")

logger.info("Downloading and reading train artifact")
local_path = run.use_artifact(artifact_input_name).file()
df_train = pd.read_csv(local_path)

# Spliting train.csv into train and validation dataset
logger.info("Spliting data into train/val")
# split-out train/validation and test dataset
x_train, x_val, y_train, y_val = train_test_split(df_train.drop(labels=stratify,axis=1),
                                                  df_train[stratify],
                                                  test_size=val_size,
                                                  random_state=seed,
                                                  shuffle=True,
                                                  stratify=df_train[stratify])
~~~

Observando os valores de treino e teste
~~~
logger.info("x train: {}".format(x_train.shape))
logger.info("y train: {}".format(y_train.shape))
logger.info("x val: {}".format(x_val.shape))
logger.info("y val: {}".format(y_val.shape))
~~~

Identificando os outliers
~~~
logger.info("Outlier Removal")
# temporary variable
x = x_train.select_dtypes("int64").copy()

# identify outlier in the dataset
lof = LocalOutlierFactor()
outlier = lof.fit_predict(x)
mask = outlier != -1
~~~

Antes e depois da retirada dos outliers
~~~
logger.info("x_train shape [original]: {}".format(x_train.shape))
logger.info("x_train shape [outlier removal]: {}".format(x_train.loc[mask,:].shape))
~~~

Conjunto de treinamento e teste após retirada de dados
~~~
# AVOID data leakage and you should not do this procedure in the preprocessing stage
# Note that we did not perform this procedure in the validation set
x_train = x_train.loc[mask,:].copy()
y_train = y_train[mask].copy()
~~~

<b> Fit e transform </b>
~~~
logger.info("Encoding Target Variable")
# define a categorical encoding for target variable
le = LabelEncoder()

# fit and transform y_train
y_train = le.fit_transform(y_train)

# transform y_test (avoiding data leakage)
y_val = le.transform(y_val)

logger.info("Classes [0, 1]: {}".format(le.inverse_transform([0, 1])))
~~~

Definição da classe <b> FeatureSelector </b>
~~~
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self.feature_names = feature_names

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what this custom transformer need to do
    def transform(self, X, y=None):
        return X[self.feature_names]
~~~

For validation purposes - object
~~~
fs = FeatureSelector(x_train.select_dtypes("object").columns.to_list())
df = fs.fit_transform(x_train)
df.head()
~~~

For validation purposes - int64
~~~
fs = FeatureSelector(x_train.select_dtypes("int64").columns.to_list())
df = fs.fit_transform(x_train)
df.head()
~~~

Lidando com os categóricos
~~~
# Handling categorical features
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes one boolean as its argument
    def __init__(self, new_features=True, colnames=None):
        self.new_features = new_features
        self.colnames = colnames

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self):
        return self.colnames.tolist()

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)

        # Remove white space in categorical features
        df = df.apply(lambda row: row.str.strip())

                # update column names
        self.colnames = df.columns

        return df
~~~

For validation purposes
~~~
fs = FeatureSelector(x_train.select_dtypes("object").columns.to_list())
df = fs.fit_transform(x_train)

ct = CategoricalTransformer(new_features=True,colnames=df.columns.tolist())
df_cat = ct.fit_transform(df)
~~~

Verifique a cardinalidade antes e depois da transformação
~~~
x_train.select_dtypes("object").apply(pd.Series.nunique)
df_cat.apply(pd.Series.nunique)
~~~


Transformar as características numéricas
~~~
class NumericalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes a model parameter as its argument
    # model 0: minmax
    # model 1: standard
    # model 2: without scaler
    def __init__(self, model=0, colnames=None):
        self.model = model
        self.colnames = colnames
        self.scaler = None

    # Fit is used only to learn statistical about Scalers
    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)
        # minmax
        if self.model == 0:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df)
        # standard scaler
        elif self.model == 1:
            self.scaler = StandardScaler()
            self.scaler.fit(df)
        return self

    # return columns names after transformation
    def get_feature_names_out(self):
        return self.colnames

    # Transformer method we wrote for this transformer
    # Use fitted scalers
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)

        # update columns name
        self.colnames = df.columns.tolist()

        # minmax
        if self.model == 0:
            # transform data
            df = self.scaler.transform(df)
        elif self.model == 1:
            # transform data
            df = self.scaler.transform(df)
        else:
            df = df.values

        return df
~~~

Validação
~~~
fs = FeatureSelector(x_train.select_dtypes("int64").columns.to_list())
df = fs.fit_transform(x_train)

nt = NumericalTransformer(model=1)
df_num = nt.fit_transform(df)
~~~

Modelos
~~~
# model = 0 (min-max), 1 (z-score), 2 (without normalization)
numerical_model = 0

# Categrical features to pass down the categorical pipeline
categorical_features = x_train.select_dtypes("object").columns.to_list()

# Numerical features to pass down the numerical pipeline
numerical_features = x_train.select_dtypes("int64").columns.to_list()

# Defining the steps for the categorical pipeline
categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(categorical_features)),
                                       ('imputer_cat', SimpleImputer(strategy="most_frequent")),
                                       ('cat_transformer', CategoricalTransformer(colnames=categorical_features)),
                                       # ('cat_encoder','passthrough'
                                       ('cat_encoder', OneHotEncoder(sparse=False, drop="first"))
                                       ]
                                )

# Defining the steps in the numerical pipeline
numerical_pipeline = Pipeline(steps=[('num_selector', FeatureSelector(numerical_features)),
                                     ('imputer_num', SimpleImputer(strategy="median")),
                                     ('num_transformer', NumericalTransformer(numerical_model, 
                                                                              colnames=numerical_features))])

# Combine numerical and categorical pieplines into one full big pipeline horizontally
full_pipeline_preprocessing = FeatureUnion(transformer_list=[('cat_pipeline', categorical_pipeline),
                                                             ('num_pipeline', numerical_pipeline)]
                                           )
~~~

Validaçãopós modelos
~~~
# for validation purposes
new_data = full_pipeline_preprocessing.fit_transform(x_train)
# cat_names is a numpy array
cat_names = full_pipeline_preprocessing.get_params()["cat_pipeline"][3].get_feature_names_out().tolist()
# num_names is a list
num_names = full_pipeline_preprocessing.get_params()["num_pipeline"][2].get_feature_names_out()
df = pd.DataFrame(new_data,columns = cat_names + num_names)
df.head()
~~~

Pipeline completo
~~~
# The full pipeline 
pipe = Pipeline(steps = [('full_pipeline', full_pipeline_preprocessing),
                         ("classifier",DecisionTreeClassifier())
                         ]
                )

# training
logger.info("Training")
pipe.fit(x_train, y_train)

# predict
logger.info("Infering")
predict = pipe.predict(x_val)

# Evaluation Metrics
logger.info("Evaluation metrics")
fbeta = fbeta_score(y_val, predict, beta=1, zero_division=1)
precision = precision_score(y_val, predict, zero_division=1)
recall = recall_score(y_val, predict, zero_division=1)
acc = accuracy_score(y_val, predict)

logger.info("Accuracy: {}".format(acc))
logger.info("Precision: {}".format(precision))
logger.info("Recall: {}".format(recall))
logger.info("F1: {}".format(fbeta))
~~~

Obtendo o resumo das métricas 
~~~
run.summary["Acc"] = acc
run.summary["Precision"] = precision
run.summary["Recall"] = recall
run.summary["F1"] = fbeta
~~~

Comparando a acurácia, precisão e recall
~~~
print(classification_report(y_val,predict))
~~~

Plot da matrix de confusão
~~~
fig_confusion_matrix, ax = plt.subplots(1,1,figsize=(7,4))
ConfusionMatrixDisplay(confusion_matrix(predict,y_val,labels=[1,0]),
                       display_labels=["yes","no"]).plot(values_format=".0f",ax=ax)

ax.set_xlabel("True Label")
ax.set_ylabel("Predicted Label")
plt.show()
~~~

Salvando as figuras no wandb
~~~
# Uploading figures
logger.info("Uploading figures")
run.log(
    {
        "confusion_matrix": wandb.Image(fig_confusion_matrix),
        # "other_figure": wandb.Image(other_fig)
    }
)
~~~

Parâmetros de importância 
~~~
# Feature importance
pipe.get_params()["classifier"].feature_importances_

# Get categorical and numerical columns names
cat_names = pipe.named_steps['full_pipeline'].get_params()["cat_pipeline"][3].get_feature_names_out().tolist()

# Get numerical column names
num_names = pipe.named_steps['full_pipeline'].get_params()["num_pipeline"][2].get_feature_names_out()
~~~

Juntar os nomes das colunas numéricas e categóricas
~~~
all_names = cat_names + num_names
~~~

Visualize all classifier plots
~~~
# For a complete documentation please see: https://docs.wandb.ai/guides/integrations/scikit
wandb.sklearn.plot_classifier(pipe.get_params()["classifier"],
                              full_pipeline_preprocessing.transform(x_train),
                              full_pipeline_preprocessing.transform(x_val),
                              y_train,
                              y_val,
                              predict,
                              pipe.predict_proba(x_val),
                              [0,1],
                              model_name='DT', feature_names=all_names)
~~~

Fechar o <b>run</b> para poder executar a proxima seção
~~~
run.finish()
~~~

global seed
~~~
seed = 41
~~~

Configuração do sweep
~~~
sweep_config = {
    # try grid or random
    "method": "random",
    "metric": {
        "name": "Accuracy",
        "goal": "maximize"
        },
    "parameters": {
        "criterion": {
            "values": ["gini","entropy"]
            },
        "splitter": {
            "values": ["random","best"]
        },
        "model": {
            "values": [0,1,2]
        },
        "random_state": {
            "values": [seed]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="decision_tree")
~~~

Função do treinamento com diferentes metricas e parâmetros
~~~
def train():
    with wandb.init() as run:

        # The full pipeline 
        pipe = Pipeline(steps = [('full_pipeline', full_pipeline_preprocessing),
                                    ("classifier",DecisionTreeClassifier())
                                    ]
                        )

        # update the parameters of the pipeline that we would like to tuning
        pipe.set_params(**{"full_pipeline__num_pipeline__num_transformer__model": run.config.model})
        pipe.set_params(**{"classifier__criterion": run.config.criterion})
        pipe.set_params(**{"classifier__splitter": run.config.splitter})
        pipe.set_params(**{"classifier__random_state": run.config.random_state})

        # training
        logger.info("Training")
        pipe.fit(x_train, y_train)

        # predict
        logger.info("Infering")
        predict = pipe.predict(x_val)

        # Evaluation Metrics
        logger.info("Evaluation metrics")
        fbeta = fbeta_score(y_val, predict, beta=1, zero_division=1)
        precision = precision_score(y_val, predict, zero_division=1)
        recall = recall_score(y_val, predict, zero_division=1)
        acc = accuracy_score(y_val, predict)

        logger.info("Accuracy: {}".format(acc))
        logger.info("Precision: {}".format(precision))
        logger.info("Recall: {}".format(recall))
        logger.info("F1: {}".format(fbeta))

        run.summary["Accuracy"] = acc
        run.summary["Precision"] = precision
        run.summary["Recall"] = recall
        run.summary["F1"] = fbeta
~~~

Testar o treinamento com diferentes metricas e parâmetros
~~~
wandb.agent(sweep_id, train, count=8)
~~~

Escolhendo o melhor pipeline após os testes anteriores. <b>Nesta etapa pode ser que você encontre características melhores </b>.  
~~~
# The full pipeline 
pipe = Pipeline(steps = [('full_pipeline', full_pipeline_preprocessing),
                         ("classifier",DecisionTreeClassifier())
                         ]
                )

# update the parameters of the pipeline that we would like to tuning
pipe.set_params(**{"full_pipeline__num_pipeline__num_transformer__model": 2})
pipe.set_params(**{"classifier__criterion": 'entropy'})
pipe.set_params(**{"classifier__splitter": 'random'})
pipe.set_params(**{"classifier__random_state": 41})



# training
logger.info("Training")
pipe.fit(x_train, y_train)

# predict
logger.info("Infering")
predict = pipe.predict(x_val)

# Evaluation Metrics
logger.info("Evaluation metrics")
fbeta = fbeta_score(y_val, predict, beta=1, zero_division=1)
precision = precision_score(y_val, predict, zero_division=1)
recall = recall_score(y_val, predict, zero_division=1)
acc = accuracy_score(y_val, predict)

logger.info("Accuracy: {}".format(acc))
logger.info("Precision: {}".format(precision))
logger.info("Recall: {}".format(recall))
logger.info("F1: {}".format(fbeta))

run.summary["Acc"] = acc
run.summary["Precision"] = precision
run.summary["Recall"] = recall
run.summary["F1"] = fbeta
~~~

Pegar as colunas numéricas e categóricas 
~~~
# Get categorical column names
cat_names = pipe.named_steps['full_pipeline'].get_params()["cat_pipeline"][3].get_feature_names_out().tolist()
# Get numerical column names
num_names = pipe.named_steps['full_pipeline'].get_params()["num_pipeline"][2].get_feature_names_out()
~~~

Juntar os nomes das colunas numéricas e categóricas
~~~
all_names = cat_names + num_names
~~~

Visualizar o melhor modelo

~~~
# Visualize all classifier plots
# For a complete documentation please see: https://docs.wandb.ai/guides/integrations/scikit
wandb.sklearn.plot_classifier(pipe.get_params()["classifier"],
                              full_pipeline_preprocessing.transform(x_train),
                              full_pipeline_preprocessing.transform(x_val),
                              y_train,
                              y_val,
                              predict,
                              pipe.predict_proba(x_val),
                              [0,1],
                              model_name='BestModel', feature_names=all_names)
~~~

Exportar o melhor modelo
~~~
# types and names of the artifacts
artifact_type = "inference_artifact"
artifact_encoder = "target_encoder"
artifact_model = "model_export"

logger.info("Dumping the artifacts to disk")
# Save the model using joblib
joblib.dump(pipe, artifact_model)

# Save the target encoder using joblib
joblib.dump(le, artifact_encoder)
~~~

Modelo do artefato
~~~
artifact = wandb.Artifact(artifact_model,
                          type=artifact_type,
                          description="A full pipeline composed of a Preprocessing Stage and a Decision Tree model"
                          )

logger.info("Logging model artifact")
artifact.add_file(artifact_model)
run.log_artifact(artifact)
~~~

Target encoder artifact
~~~
artifact = wandb.Artifact(artifact_encoder,
                          type=artifact_type,
                          description="The encoder used to encode the target variable"
                          )

logger.info("Logging target enconder artifact")
artifact.add_file(artifact_encoder)
run.log_artifact(artifact)
~~~

Finalizando etapa de treinamento

~~~
run.finish()
~~~

## 4.12 Teste
Importando algumas bibliotecas. Não se preocupe caso alguma já tenha sido importada. 

~~~
import logging
import pandas as pd
import wandb
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
~~~

Definindo as classes base
* É necessário verificar alguns importes da etapa do treinamento para utilizar o joblib.load().
~~~
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self.feature_names = feature_names

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what this custom transformer need to do
    def transform(self, X, y=None):
        return X[self.feature_names]

# Handling categorical features
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes one boolean as its argument
    def __init__(self, new_features=True, colnames=None):
        self.new_features = new_features
        self.colnames = colnames

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self):
        return self.colnames.tolist()

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)

        # Remove white space in categorical features
        df = df.apply(lambda row: row.str.strip())

        # update column names
        self.colnames = df.columns

        return df
        
# transform numerical features
class NumericalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes a model parameter as its argument
    # model 0: minmax
    # model 1: standard
    # model 2: without scaler
    def __init__(self, model=0, colnames=None):
        self.model = model
        self.colnames = colnames
        self.scaler = None

    # Fit is used only to learn statistical about Scalers
    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)
        # minmax
        if self.model == 0:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df)
        # standard scaler
        elif self.model == 1:
            self.scaler = StandardScaler()
            self.scaler.fit(df)
        return self

    # return columns names after transformation
    def get_feature_names_out(self):
        return self.colnames

    # Transformer method we wrote for this transformer
    # Use fitted scalers
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)

        # update columns name
        self.colnames = df.columns.tolist()

        # minmax
        if self.model == 0:
            # transform data
            df = self.scaler.transform(df)
        elif self.model == 1:
            # transform data
            df = self.scaler.transform(df)
        else:
            df = df.values

        return df
~~~

Avaliação


~~~
# global variables

# name of the artifact related to test dataset
artifact_test_name = "decision_tree/test.csv:latest"

# name of the model artifact
artifact_model_name = "decision_tree/model_export:latest"

# name of the target encoder artifact
artifact_encoder_name = "decision_tree/target_encoder:latest"
~~~

Configuar logging
~~~
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()
~~~

Iniciar o projeto no wandb
~~~
run = wandb.init(project="decision_tree",job_type="test")
~~~

~~~
logger.info("Downloading and reading test artifact")
test_data_path = run.use_artifact(artifact_test_name).file()
df_test = pd.read_csv(test_data_path)

# Extract the target from the features
logger.info("Extracting target from dataframe")
x_test = df_test.copy()
y_test = x_test.pop("y")
~~~

Extract the encoding of the target variable
~~~
logger.info("Extracting the encoding of the target variable")
encoder_export_path = run.use_artifact(artifact_encoder_name).file()
le = joblib.load(encoder_export_path)
~~~

Transform y_train
~~~
y_test = le.transform(y_test)
logger.info("Classes [0, 1]: {}".format(le.inverse_transform([0, 1])))
~~~

Download do artefato de inferência
~~~
logger.info("Downloading and load the exported model")
model_export_path = run.use_artifact(artifact_model_name).file()
pipe = joblib.load(model_export_path)
~~~


Predição
~~~
logger.info("Infering")
predict = pipe.predict(x_test)
~~~

Avaliação das Métricas
~~~
logger.info("Test Evaluation metrics")
fbeta = fbeta_score(y_test, predict, beta=1, zero_division=1)
precision = precision_score(y_test, predict, zero_division=1)
recall = recall_score(y_test, predict, zero_division=1)
acc = accuracy_score(y_test, predict)

logger.info("Test Accuracy: {}".format(acc))
logger.info("Test Precision: {}".format(precision))
logger.info("Test Recall: {}".format(recall))
logger.info("Test F1: {}".format(fbeta))

run.summary["Acc"] = acc
run.summary["Precision"] = precision
run.summary["Recall"] = recall
run.summary["F1"] = fbeta
~~~

Comparando a acurácia, precisão e recall
~~~
print(classification_report(y_test,predict))
~~~

Matrix de confusão
~~~
fig_confusion_matrix, ax = plt.subplots(1,1,figsize=(7,4))
ConfusionMatrixDisplay(confusion_matrix(predict,y_test,labels=[1,0]),
                       display_labels=["yes","no"]).plot(values_format=".0f",ax=ax)

ax.set_xlabel("True Label")
ax.set_ylabel("Predicted Label")
plt.show()
~~~

Uploading figures
~~~
logger.info("Uploading figures")
run.log(
    {
        "confusion_matrix": wandb.Image(fig_confusion_matrix),
        # "other_figure": wandb.Image(other_fig)
    }
)
~~~

Finalizando
~~~
run.finish()
~~~
