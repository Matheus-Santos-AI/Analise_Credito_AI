import pandas as pd
from pathlib import Path

#Busca de documentos no repositorio
BASE_DIR = Path(__file__).resolve().parent.parent

#Caregando dados historicos de crédito
df_historico_credito = pd.read_csv(BASE_DIR / "doc/historico_credito.csv")


pd.set_option('display.max_columns', None)

#Carregando dados de solicitações de creditos simulados.
df_solicitacoes = pd.read_csv("../doc/dados_pessoa_fisica.csv")


#Tratando dados de CPF (removendo pontuações e espaços)
cpf = df_solicitacoes['cpf'].str.replace(r"\D", "", regex=True)

#Criando uma indexação para o CPF
id_cpf = cpf.str[:3] + cpf.str[-2:]

#Criando uma indexação para o nome utilizando as iniciais
id_nome = (
    df_solicitacoes['nome']
    .fillna("")
    .apply(lambda s: "".join(p[0] for p in s.split()))
    .str.upper()
)

#Criando um ID unico com a junção dos dados de cpf + iniciais
df_solicitacoes['id_pessoa'] = id_nome + id_cpf







#Removendo colunas não necessárias e com dados sensiveis
df_solicitacoes = df_solicitacoes.drop(columns =["cpf", "nome", "sexo"], errors= "ignore")

#Treinando o modelo de Machine Learning RANDOMFORESTREGRESSOR

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

#Separando variavel alvo e variaveis explicativas
X = df_historico_credito.drop(columns=["valor_aprovado"])
y = df_historico_credito["valor_aprovado"]

#Seprando dados em treino e teste na proporção 20/80
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Transformando strings em float para o treinamento
preprocessador = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["profissao", "bairro"]), #transformando em vetor
        ("ordinal", OrdinalEncoder(
            categories=[["Nenhuma", "Aval", "Veículo", "Aplicação financeira", "Imóvel"]]  #transformando em numeros inteiros com hierarquia
        ), ["garantia"]),
    ],
    remainder= StandardScaler(), #As colunas não mencionadas serão aplicadas o metodo StandadScaler
)

#Criando fluxo de pre´-processamento e modelagem do aprendizado
modelo = Pipeline([
    ("prep", preprocessador),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
])

#Teinamento do modelo RandomForestRegressor
modelo.fit(X_train, y_train)

r2_ml = modelo.score(X_test, y_test)
#Teste
print("R²:", r2_ml)

#Predict das solicitações de credito
X_novo = df_solicitacoes[X_train.columns]


previsoes = modelo.predict(X_novo)


df_solicitacoes["valor_previsto"] = previsoes


#Testando dados manualmente para comparação

df_teste = pd.DataFrame([{
    "idade": 35,
    "profissao": "Analista financeiro",
    "renda_mensal": 6500.00,
    "tempo_emprego_meses": 48,
    "comprometimento_renda": 0.25,
    "divida_ativa": 30000.00,
    "score": 720,
    "qtd_atrasos_24m": 0,
    "max_dias_atraso": 0,
    "tempo_relacionamento_meses": 36,
    "tempo_residencia_anos": 5,
    "bairro": "Mocca",
    "qtd_filhos": 1,
    "garantia": "Veículo",
}])

X_teste = df_teste[X_train.columns]      # mesmas colunas e mesma ordem do treino
valor_ml = float(modelo.predict(X_teste)[0])

print(f"Valor liberado: R${valor_ml: ,.2f}")

#=======================================================================================================================





#MODELO SVR
from sklearn.svm import SVR


modelo_svr = Pipeline([
    ("prep", preprocessador),
    ("svr", SVR(kernel = "poly", C=90, epsilon=0.01, gamma = "scale")),
])

modelo_svr.fit(X_train, y_train)

r2_svr = modelo_svr.score(X_test, y_test)

print("R2:", r2_svr)

X_novo_svr = df_solicitacoes[X_train.columns]

previsoes_svr = modelo_svr.predict(X_novo)

x_teste_svr = df_teste[X_train.columns]
valor_svr = float(modelo_svr.predict(X_teste)[0])

print(valor_svr)









#=======================================================================================================================
#CRIANDO MODELO DE REDES NEURAIS PARA O MESMO CASO
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


cat_cols = ["profissao", "bairro"]
ord_cols = ["garantia"]
num_cols = [c for c in X_train.columns if c not in cat_cols + ord_cols]

prep = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("ordinal", Pipeline([
        ("enc", OrdinalEncoder(categories=[["Nenhuma", "Aval", "Veículo", "Aplicação financeira", "Imóvel"]])),
        ("sc", StandardScaler()),
    ]), ord_cols),
    ("num", StandardScaler(), num_cols),
])

X_train_p = prep.fit_transform(X_train)    # ajusta SÓ no treino
X_test_p = prep.transform(X_test)

y_train_log = np.log1p(y_train) #Normaliza os numeros com uma transformação logaritmica para que tenha uma distribuição normal entre eles , ajuda na regressão logistica

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


tf.keras.utils.set_random_seed(42)

#Modelos utilizados para treinamento , camadas
model = keras.Sequential([
    layers.Input(shape=(X_train_p.shape[1],)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(64, activation="swish"),
    layers.BatchNormalization(),
    layers.Dense(32, activation="swish"),
    layers.Dense(32, activation="elu"),
    layers.Dense(16, activation="elu"),
    layers.Dense(16, activation="relu"),

    layers.Dense(32, activation="relu"),
    layers.Dense(1),                      # saída linear (regressão)
])

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])  #otimizador (Adam) , taxa de aprendizado (1e-3) , função de perda (mse) , monitoramente de desempenho (mae)

callbacks = [
    keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True), #para o treinamento caso as metricas não melhorem em 15 epocas , restore_best_weights=True pega o peso da epoca que demosntrou melhor desempenho
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5), #Reduz a taxa de aprendizado automaticamente quando a métrica monitorada não melhora, factor=0.5 → divide a taxa de aprendizado por 2 , patience=5 → espera 5 épocas sem melhora antes de reduzir
]

#Treinamento
history = model.fit(
    X_train_p, y_train_log,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=0,
)

from sklearn.metrics import mean_absolute_error, r2_score

#Predict de valores
pred = np.expm1(model.predict(X_test_p).ravel())    # volta para R$ revertendo a transformação logaritmica feita anteriormente
pred = np.where(pred < 500, 0, pred)                # mínimo aprovado no dataset
pred = np.clip(pred, 0, None)

r2_dp = r2_score(y_test, pred)
print("MAE (R$):", mean_absolute_error(y_test, pred))
print("R²:", r2_dp)

import numpy as np
import pandas as pd

def prever(df_novo):
    # mesmas colunas e mesma ordem do treino
    X_novo = df_novo[X_train.columns]

    # mesmo pré-processador do treino (só transform, nunca fit)
    X_novo_p = prep.transform(X_novo)

    # o modelo foi treinado em log1p, então volta com expm1
    valor = np.expm1(model.predict(X_novo_p, verbose=0).ravel())
    valor = np.where(valor < 500, 0, valor)      # mínimo aprovado no dataset
    valor = np.round(valor, -2)                  # arredonda resultado

    resultado = df_novo.copy()
    resultado["valor_previsto"] = valor
    return resultado

#Teste com mesmo exemplo aplicado no modelo anterior de Machine Learning

valor = prever(df_teste)["valor_previsto"].iloc[0]





print(f"Valor liberado com Modelo de Machine Learn : R${valor_ml: ,.2f}")
print(f"Valor Liberado com Modelo de Deep Learning: R$ {valor:,.2f}")


resultado_comparativo = pd.DataFrame([
    {"Modelo": "RF", "Valor Liberado": valor_ml, "R2": r2_ml},
    {"Modelo": "DP", "Valor Liberado": valor,    "R2": r2_dp},
    {"Modelo": "SCR", "Valor Liberado": valor_svr, "R2": r2_svr}

])

print(resultado_comparativo)




