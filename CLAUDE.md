# TechChallenger3 — Previsão de Atrasos de Voos (EUA)

## Visão Geral do Projeto

Pipeline completo de ciência de dados para análise e predição de atrasos de voos nos EUA. Envolve EDA, modelagem supervisionada (classificação/regressão) e não supervisionada (clustering/PCA), com apresentação crítica de resultados.

**Instituição:** FIAP  
**Dataset:** Voos nos EUA (5,8 milhões de registros, 31 colunas)

---

## Stack Tecnológica

- **Linguagem:** Python 3.10+
- **Notebooks:** Jupyter (`.ipynb` em `notebooks/`)
- **Dados:** Pandas, NumPy
- **Visualização:** Matplotlib, Seaborn, Plotly
- **ML Supervisionado:** Scikit-learn, XGBoost, LightGBM
- **ML Não Supervisionado:** Scikit-learn (KMeans, DBSCAN, PCA)
- **Geografico (opcional):** Folium, GeoPandas

---

## Estrutura de Diretórios

```
TechChallenger3/
├── CLAUDE.md                  # Este arquivo
├── espec-proj.md              # Especificação oficial do projeto
├── planejamento.md            # Notas de planejamento
├── dados/                     # Dados brutos (NÃO modificar)
│   ├── flights.csv            # 5.8M voos, 31 colunas (565 MB)
│   ├── airports.csv           # Aeroportos (IATA, coords)
│   └── airlines.csv           # Companhias aéreas
├── notebooks/                 # Jupyter notebooks por fase
│   ├── 01_eda.ipynb
│   ├── 02_supervisionado.ipynb
│   ├── 03_nao_supervisionado.ipynb
│   └── 04_resultados.ipynb
├── src/                       # Código Python reutilizável
│   ├── data/                  # Carregamento e pré-processamento
│   ├── models/                # Treinamento e avaliação
│   └── visualization/         # Funções de gráficos
├── outputs/
│   ├── figures/               # Gráficos exportados
│   └── models/                # Modelos serializados (.pkl, .joblib)
└── .claude/
    └── commands/              # Slash commands customizados
```

---

## Dados

### flights.csv — Colunas Principais

| Grupo | Colunas |
|-------|---------|
| Temporal | YEAR, MONTH, DAY, DAY_OF_WEEK |
| Voo | AIRLINE, FLIGHT_NUMBER, TAIL_NUMBER |
| Rota | ORIGIN_AIRPORT, DESTINATION_AIRPORT, DISTANCE |
| Horários | SCHEDULED_DEPARTURE, DEPARTURE_TIME, SCHEDULED_ARRIVAL, ARRIVAL_TIME |
| **Targets** | **DEPARTURE_DELAY, ARRIVAL_DELAY** |
| Causas | AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY, WEATHER_DELAY |
| Status | DIVERTED, CANCELLED, CANCELLATION_REASON |

### Variável Target Recomendada
- **Classificação:** `ARRIVAL_DELAY > 15` → binário (atrasado sim/não)
- **Regressão:** `ARRIVAL_DELAY` contínuo (minutos)

---

## Fases do Projeto

### Fase 1 — EDA (`/eda`)
- Estatísticas descritivas de todas as colunas
- Distribuição de atrasos por airline, aeroporto, dia da semana, mês
- Tratamento de missing values (CANCELLATION_REASON, delay columns)
- Visualizações: histogramas, boxplots, heatmap de correlação

### Fase 2 — Supervisionado (`/train`)
- Feature engineering (período do dia, flag de feriado, season)
- Classificação: LogReg, Random Forest, XGBoost (comparar métricas)
- Regressão: LinearReg, GBM (comparar RMSE, MAE, R²)
- Cross-validation e análise de feature importance

### Fase 3 — Não Supervisionado (`/cluster`)
- KMeans em aeroportos por perfil de atraso
- PCA para redução dimensional
- Interpretação dos clusters com visualizações

### Fase 4 — Resultados (`/report`)
- Consolidar principais achados
- Limitações dos modelos
- Próximos passos e melhorias

---

## Comandos Comuns

```bash
# Instalar dependências
pip install -r requirements.txt

# Iniciar Jupyter
jupyter lab notebooks/

# Executar pipeline completo
python src/pipeline.py
```

---

## Convenções

- Notebooks numerados por fase: `01_eda.ipynb`, `02_supervisionado.ipynb`
- Funções reutilizáveis vão em `src/`, importadas nos notebooks
- Gráficos salvos em `outputs/figures/` com nome descritivo
- Modelos salvos em `outputs/models/` com versão e data
- Dados em `dados/` são **somente leitura** — nunca modificar os CSVs originais

---

## Perguntas Guia

1. Quais aeroportos são mais críticos em relação a atrasos?
2. Que características aumentam a chance de atraso?
3. Atrasos são mais comuns em certos dias/horários?
4. É possível agrupar aeroportos com perfis semelhantes?
