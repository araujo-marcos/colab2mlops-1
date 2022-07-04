# Introdução

O projeto aborda um projeto de Machine Learning com o intuito de prever o sucesso de chamadas de telemarketing para a venda de depósitos bancários de longo prazo (disponível no link: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#). Foi abordado como fonte os dados de um banco português, com dados recolhidos de 2008 a 2013, incluindo dessa forma os efeitos da crise financeira vivida mundialmente a partir de 2009. A análise efetuada confirmou o modelo obtido como credível e valioso para os gestores de campanhas de telemarketing.

As campanhas de marketing foram baseadas em telefonemas e muitas vezes era necessário mais de um contato com o mesmo cliente para ter noção se o produto em análise (depósito bancário a prazo) seria ou não efetuado.

No banco de dados utilizado no projeto existe um conjunto de dados:
bank-full.csv: com todos os exemplos e 17 entradas, ordenadas por data (versão mais antiga deste conjunto de dados com menos entradas).

O objetivo da classificação é prever se o cliente irá subscrever um depósito a prazo (variável y).

Para o desenvolvimento deste estudo foram utilizadas algumas tecnologias, como Marchin Learn, K-means, Decision Tree, Pipeline e Weights & Biases.

# Informação dos Atributos

A seguir serão apresentadas as informações dos atributos utilizados no projeto, detalhando, inclusive, a categorização de cada variável.

## Variáveis de Entrada

### Dados do cliente do banco:

* age: idade (numérico); 
* job: tipo de emprego (administrador, colarinho azul, empreendedor, empregada doméstica, gerenciamento, aposentado, autônomo, serviços, estudante, técnico, desempregado, desconhecido); 
* marital: estado civil (divorciado, casado, solteiro, desconhecido, nota: divorciado significa divorciado ou viúvo); 
* education: escolaridade (básico.4anos, básico.6 anos, básico.9 anos, ensino médio, analfabetos, curso profissional, grau universitário, desconhecido); 
* default: tem crédito inadimplente? (não, sim, desconhecido); 
* housing: tem crédito de habitação? (não, sim, desconhecido); 
* loan: tem empréstimo pessoal? (não, sim, desconhecido);

### Dados relacionados com o último contato da campanha atual:

* contact: tipo de comunicação do contato (celular, telefone);
* month: último mês de contato do ano (jan, fev, mar, ..., nov, dec); 
* day of week: último dia de contato da semana (seg, ter, qua, qui, sex); 
* duration: duração do último contato, em segundos (numérico). Observação importante: esse atributo afeta muito o destino de saída (por exemplo, se duração = 0, então y = não). No entanto, a duração não é conhecida antes de uma chamada ser realizada. Além disso, após o término da chamada, y é obviamente conhecido.

### Outros atributos:
* campaign: número de contatos realizados durante esta campanha e para este cliente (numérico, inclui último contato); 
* pdays: número de dias que se passaram após o último contato com o cliente de uma campanha anterior (numérico; 999 significa que o cliente não foi contatado anteriormente); 
* previous: número de contatos realizados antes desta campanha e para este cliente (numérico); 
* Poutcome: resultado da campanha de marketing anterior (fracasso, inexistente, sucesso);

###  Atributos do contexto social e econômico
* emp.var.rate: taxa de variação do emprego - indicador trimestral (numérico); 
* cons.price.idx: índice de preços ao consumidor - indicador mensal (numérico); 
* cons.conf.idx: índice de confiança do consumidor - indicador mensal (numérico); 
* euribor3m: taxa de 3 meses euribor - indicador diário (numérico); 
* nr.employed: número de funcionários - indicador trimestral (numérico).

## Variáveis de Saída
* y - o cliente realizou um depósito a prazo? (sim, não).

# Detalhes do <i> Workflow </i>
  
  Para a execução do projeto foram utilizados os passos apresentados no workflow criado e disponibilizado pelo Professor Ivanovitch Silva. Os passos desse pipeline foram conforme a imagem abaixo.
  
  ![Workflow](https://github.com/ivanovitchm/ppgeecmachinelearning/blob/main/images/workflow.png)

# Instalando as bibliotecas:


  
