# 💳 Previsão de Crédito Aprovado com Machine Learning e Deep Learning

Projeto de ciência de dados que estima o **valor de crédito a ser liberado** para pessoas físicas que residem na cidade de São Paulo, a partir do perfil financeiro e pessoal do solicitante. O mesmo problema de regressão é resolvido com três abordagens diferentes, para que seus desempenhos possam ser comparados:

| Sigla | Modelo | Biblioteca |
|-------|--------|------------|
| **RF** | Random Forest Regressor | scikit-learn |
| **SVR** | Support Vector Regression (kernel polinomial) | scikit-learn |
| **DP** | Rede Neural Densa (Deep Learning) | TensorFlow / Keras |

---

## 📌 Objetivo

Com base em um histórico de créditos já concedidos (`historico_credito.csv`), treinar modelos capazes de prever a coluna `valor_aprovado` e aplicá-los a novas solicitações simuladas (`dados_pessoa_fisica.csv`), indicando o valor liberado ou a negativa do crédito (valor = 0).

---

## 🗂️ Estrutura do repositório

```
.
├── doc/
│   ├── historico_credito.csv        # Base histórica simulada usada no treino/teste
│   └── dados_pessoa_fisica.csv      # Solicitações simuladas a serem previstas
├── src/
│   └── main.py                      # Script principal 
├── requirements.txt
└── README.md
```

> O script resolve a raiz do projeto com `Path(__file__).resolve().parent.parent`, ou seja, ele deve ficar **uma pasta abaixo** da raiz (ex.: `src/`), com a pasta `doc/` na raiz.

---

## 🧾 Variáveis utilizadas

**Variável alvo:** `valor_aprovado`

**Variáveis explicativas:**

| Variável | Tipo | Descrição |
|----------|------|-----------|
| `idade` | numérica | Idade do solicitante |
| `profissao` | categórica | Profissão |
| `renda_mensal` | numérica | Renda mensal (R$) |
| `tempo_emprego_meses` | numérica | Tempo no emprego atual |
| `comprometimento_renda` | numérica | Percentual da renda já comprometido |
| `divida_ativa` | numérica | Valor de dívidas ativas (R$) |
| `score` | numérica | Score de crédito |
| `qtd_atrasos_24m` | numérica | Quantidade de atrasos nos últimos 24 meses |
| `max_dias_atraso` | numérica | Maior atraso registrado (dias) |
| `tempo_relacionamento_meses` | numérica | Tempo de relacionamento com a instituição |
| `tempo_residencia_anos` | numérica | Tempo na residência atual |
| `bairro` | categórica | Bairro de residência |
| `qtd_filhos` | numérica | Número de filhos |
| `garantia` | ordinal | `Nenhuma` < `Aval` < `Veículo` < `Aplicação financeira` < `Imóvel` |

---

## ⚙️ Fluxo do projeto

### 1. Preparação dos dados e privacidade
- Leitura dos CSVs com `pandas`.
- Limpeza do CPF (remoção de pontuação e espaços).
- Criação de um **identificador anônimo** (`id_pessoa`) combinando as **iniciais do nome** com os **3 primeiros e 2 últimos dígitos do CPF**.
- Remoção das colunas sensíveis (`cpf`, `nome`, `sexo`) antes da modelagem, em linha com boas práticas de proteção de dados (LGPD).

### 2. Pré-processamento
- `OneHotEncoder` para `profissao` e `bairro`.
- `OrdinalEncoder` com hierarquia definida para `garantia`.
- `StandardScaler` para as variáveis numéricas.
- Divisão **80% treino / 20% teste** (`random_state=42`).

### 3. Modelos

**Random Forest** — `RandomForestRegressor(n_estimators=100)` dentro de um `Pipeline` com o pré-processador.

**SVR** — `SVR(kernel="poly", C=90, epsilon=0.01, gamma="scale")` no mesmo formato de `Pipeline`.

**Rede Neural (Keras)**
- Alvo transformado com `log1p` e revertido com `expm1` na predição.
- Arquitetura densa com ativações `relu`, `swish` e `elu`, além de `Dropout` e `BatchNormalization`.
- Otimizador Adam (`1e-3`), perda `mse`, métrica `mae`.
- Callbacks: `EarlyStopping` (paciência 15) e `ReduceLROnPlateau` (fator 0,5, paciência 5).
- Pós-processamento: valores abaixo de R$ 500 (mínimo aprovado no dataset) são convertidos para 0 (crédito negado) e o resultado é arredondado para a centena.

### 4. Avaliação e comparação
Os modelos são avaliados com **R²** (e **MAE** para a rede neural) no conjunto de teste e comparados em uma tabela final, junto com a predição de um cliente de exemplo:

```python
{
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
}
```

---

## 🚀 Como executar

### Pré-requisitos
- Python 3.9+

### Instalação

```bash
# Clonar o repositório
git clone <url-do-repositorio>
cd <nome-do-repositorio>

# (Opcional) criar ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

Exemplo de `requirements.txt`:

```
pandas
numpy
scikit-learn
tensorflow
```

### Execução

```bash
cd src
python main.py
```

> Execute a partir da pasta do script: a leitura de `dados_pessoa_fisica.csv` usa o caminho relativo `../doc/`.

---

## 📊 Resultados

Ao final da execução, o script imprime uma tabela comparativa neste formato:

| Modelo | Valor Liberado (cliente exemplo) | R² (teste) |
|--------|----------------------------------|------|
| RF  | 13948.00 | 0.85 |
| DP  |  9200.00 | 0.82|
| SCR | 7085.42 | 0.31 |



---


## 🛠️ Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)

---

## ⚠️ Aviso

Todos os dados desse projeto são **dados simulados** e tem somente finalidade **educacional**. Não deve ser usado como única base para decisões reais de concessão de crédito.

## 👤 Autor

Desenvolvido por **Matheus Santos** — [LinkedIn](linkedin.com/in/matheussantos93/) | [GitHub](https://github.com/Matheus-Santos-AI)