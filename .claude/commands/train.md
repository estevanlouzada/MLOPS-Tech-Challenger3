# /train — Modelagem Supervisionada

Execute a Fase 2 do TechChallenger3: feature engineering e treinamento de modelos supervisionados.

## Pré-requisito

Fase 1 (EDA) concluída. Resultados confirmados pelo notebook `Interactive-eda.ipynb`:
- **5.714.008 voos** completados (base de treino)
- **17,91% atrasados** > 15 min → dataset desbalanceado (82% pontual / 18% atrasado)
- Mediana ARRIVAL_DELAY: **−5 min** (maioria chega adiantada)
- AIRLINE: **14 únicos** | ORIGIN_AIRPORT: **628** | DESTINATION_AIRPORT: **629**

---

## Aviso Crítico — Data Leakage

As seguintes colunas **NÃO podem ser usadas como features** em um modelo preditivo real
(dados disponíveis apenas após o voo):

| Coluna | Motivo |
|--------|--------|
| `DEPARTURE_DELAY` | r = 0.96 com ARRIVAL_DELAY — leakage direto |
| `DEPARTURE_TIME` | horário real de partida (desconhecido antes do voo) |
| `TAXI_OUT` / `TAXI_IN` | operacional pós-partida |
| `ELAPSED_TIME` / `AIR_TIME` | tempo real de voo |
| `AIR_SYSTEM_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`, `SECURITY_DELAY` | causas do atraso (só existem depois do fato) |
| `TIME_RECOVERED` | derivado de DEPARTURE_DELAY |

**Regra:** o modelo deve funcionar apenas com dados conhecidos **antes** do embarque.

---

## Feature Engineering

### Targets

- `IS_DELAYED = (ARRIVAL_DELAY > 15).astype(int)` → classificação binária (17,91% positivos)
- `ARRIVAL_DELAY` → regressão (distribuição assimétrica, cauda longa até 1971 min)

### Features Derivadas da EDA

Criar a partir de dados disponíveis antes do voo:

```python
# Temporal — Insight 1 (efeito cascata) e Insight 2 (sazonalidade)
df["DEPARTURE_HOUR"]   = df["SCHEDULED_DEPARTURE"] // 100  # 0–23
df["DEPARTURE_PERIOD"] = pd.cut(df["DEPARTURE_HOUR"],
                                bins=[-1, 5, 11, 17, 23],
                                labels=["Madrugada(0-5)", "Manhã(6-11)",
                                        "Tarde(12-17)", "Noite(18-23)"])
df["SEASON"] = df["MONTH"].map({12:"Inverno",1:"Inverno",2:"Inverno",
                                 3:"Primavera",4:"Primavera",5:"Primavera",
                                 6:"Verão",7:"Verão",8:"Verão",
                                 9:"Outono",10:"Outono",11:"Outono"})
df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 6).astype(int)  # Insight 9: sexta crítica

# Rota — Insight 6 (distância) + Seção 7 (rotas problemáticas)
df["ROUTE"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)
df["DISTANCE_BUCKET"] = pd.cut(df["DISTANCE"],
                               bins=[0, 500, 1500, 10_000],
                               labels=["Curto", "Medio", "Longo"])

# Buffer de schedule — pontualidade operacional (Seção 4)
# SCHEDULED_TIME já existe — manter como feature numérica
```

### Features Finais para o Modelo

| Feature | Tipo | Origem | Justificativa EDA |
|---------|------|--------|-------------------|
| `MONTH` | Numérica | Direto | Sazonalidade (Insight 2) |
| `DAY_OF_WEEK` | Numérica | Direto | Sexta crítica (Insight 9) |
| `IS_WEEKEND` | Binária | Derivada | Insight 9 |
| `DEPARTURE_HOUR` | Numérica | Derivada | Efeito cascata (Insight 1) |
| `DEPARTURE_PERIOD` | Categórica | Derivada | Agrupamento do DEPARTURE_HOUR |
| `SEASON` | Categórica | Derivada | Verão/inverno (Insight 2) |
| `AIRLINE` | Categórica | Direto | 14 únicos — comportamento diferenciado (Insight 4) |
| `ORIGIN_AIRPORT` | Categórica | Direto | 628 únicos — aeroportos problemáticos (Seção 6) |
| `DESTINATION_AIRPORT` | Categórica | Direto | 629 únicos — destinos atrasados (Seção 6) |
| `ROUTE` | Categórica | Derivada | Rotas com maior atraso (Seção 7) |
| `DISTANCE` | Numérica | Direto | Voos curtos atrasam mais (Insight 6) |
| `DISTANCE_BUCKET` | Categórica | Derivada | Insight 6 |
| `SCHEDULED_TIME` | Numérica | Direto | Buffer de pontualidade (Seção 4) |
| `SCHEDULED_DEPARTURE` | Numérica | Direto | Horário planejado |

---

## Preparação dos Dados

```python
# 1. Partir do flights_ok da EDA (cancelados e desviados já removidos)
#    5.714.008 registros

# 2. Remover colunas com leakage (ver tabela acima)
LEAKAGE_COLS = ["DEPARTURE_DELAY","DEPARTURE_TIME","TAXI_OUT","TAXI_IN",
                "ELAPSED_TIME","AIR_TIME","WHEELS_OFF","WHEELS_ON",
                "AIR_SYSTEM_DELAY","SECURITY_DELAY","AIRLINE_DELAY",
                "LATE_AIRCRAFT_DELAY","WEATHER_DELAY","TIME_RECOVERED",
                "ARRIVAL_TIME"]

# 3. Encoding
# AIRLINE (14 únicos) → One-Hot Encoding (pd.get_dummies ou OneHotEncoder)
# ORIGIN_AIRPORT / DESTINATION_AIRPORT / ROUTE (alta cardinalidade) → Target Encoding
#   usar category_encoders.TargetEncoder ou mean encoding manual no treino
# DEPARTURE_PERIOD / SEASON / DISTANCE_BUCKET → Ordinal ou One-Hot

# 4. Split estratificado (manter proporção 17.91% positivos)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_clf
)

# 5. Scaling apenas para modelos lineares (LogReg)
#    Tree-based (RF, XGBoost) não precisam de scaling
```

### Tratamento do Desbalanceamento (17,91% positivos)

Usar pelo menos **uma** das estratégias:
- `class_weight="balanced"` nos modelos sklearn (LogReg, RF)
- `scale_pos_weight = (1 - 0.1791) / 0.1791 ≈ 4.58` no XGBoost
- Avaliar F1 e ROC-AUC como métricas principais (accuracy é enganosa com dados desbalanceados)

---

## Classificação — Prever IS_DELAYED (>15 min)

Treinar e comparar os 3 modelos abaixo. **Priorizar F1 e ROC-AUC** sobre accuracy.

### 1. Logistic Regression (baseline)
```python
LogisticRegression(class_weight="balanced", max_iter=500, C=1.0)
```
- Baseline interpretável
- Requer StandardScaler nas features numéricas

### 2. Random Forest
```python
RandomForestClassifier(n_estimators=200, max_depth=15,
                       class_weight="balanced", random_state=42, n_jobs=-1)
```
- Captura interações não-lineares (DEPARTURE_HOUR × AIRLINE × ORIGIN)
- Gerar feature importance

### 3. XGBoost
```python
XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
              scale_pos_weight=4.58, random_state=42,
              eval_metric="auc", early_stopping_rounds=20)
```
- Usar validação interna com eval_set para early stopping
- Espera-se ser o melhor modelo

### Métricas e Plots

```python
# Para cada modelo:
print(classification_report(y_test, y_pred))
# Tabela: Precision, Recall, F1, ROC-AUC, Accuracy

# Plots:
# - Confusion Matrix (heatmap)
# - ROC Curve (os 3 modelos sobrepostos)
# - Feature Importance (Top 15 — apenas RF e XGBoost)
# - Precision-Recall Curve (importante com desbalanceamento)
```

### Cross-Validation
```python
StratifiedKFold(n_splits=5)
# Reportar: média ± std do F1-score (fold por fold)
```

---

## Regressão — Prever ARRIVAL_DELAY (minutos)

**Atenção:** distribuição muito assimétrica (outliers até 1971 min). Considerar:
- Treinar com `ARRIVAL_DELAY.clip(-60, 360)` para reduzir impacto de outliers extremos
- Avaliar RMSE, MAE e R² — MAE é mais robusto a outliers aqui

### 1. Linear Regression (baseline)
```python
LinearRegression()  # com StandardScaler
```

### 2. Gradient Boosting Regressor
```python
GradientBoostingRegressor(n_estimators=200, max_depth=5,
                           learning_rate=0.1, random_state=42)
```
- Alternativa: `XGBRegressor` com os mesmos parâmetros

### Métricas e Plots
```python
# Para cada modelo:
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

# Plots:
# - Predicted vs Actual (scatter com linha ideal y=x)
# - Residual Plot (resíduos vs predito — detectar heterocedasticidade)
# - Distribuição dos resíduos (histograma)
```

---

## Output Esperado

- Script `notebooks/02_supervisionado.py` (padrão `# %%`)
- Modelos salvos em `outputs/models/`:
  - `clf_logreg.joblib`, `clf_rf.joblib`, `clf_xgb.joblib`
  - `reg_linear.joblib`, `reg_gbm.joblib`
- `outputs/model_comparison.csv` — tabela comparativa de métricas
- Top 15 features mais importantes (RF + XGBoost)
- Identificação do melhor modelo com justificativa baseada em F1 e ROC-AUC


## Padrao para script Python Analiticos 

Todo script Python de analise de dados deve usar delimitadores de celula `# %%`, tornando-o executavel celula por celula no VS Code Python Interactive Window (igual a um notebook, mas em .py).

* `# %%` -- inicia uma celula de codigo. O VS Code exibe "Run Cell" acima de cada uma.
* `# %% [markdown]` -- celula de texto/markdown (titulos, descricoes de secao)

O arquivo continua funcionando normalmente como script via python arquivo.py


## Regras de execução 

Primeiramente verificar as analises criadas em notebooks/01_eda.py de expliração dos dados para entender features de treinamento.
E o notebooks/Interactive-eda.ipynb e as imagens de saida que estão no caminho ../../outputs/figures