# Plano de Execução — TechChallenger3

**Data de início:** 2026-04-20  
**Objetivo:** Pipeline completo de ciência de dados para previsão de atrasos de voos (EUA)

---

## Fase 0 — Setup (1 dia)

- [x] Estrutura de diretórios criada (`notebooks/`, `src/`, `outputs/`)
- [x] CLAUDE.md criado com documentação do projeto
- [x] Slash commands configurados (`/eda`, `/train`, `/cluster`, `/report`)
- [x] `requirements.txt` instalado
- [x] Git inicializado com `.gitignore` adequado
- [x] Ambiente virtual criado (`venv` ou `conda`)

---

## Fase 1 — EDA (2-3 dias)

**Notebook:** `notebooks/01_eda.ipynb`  
**Comando:** `/eda`

### Tarefas

- [x] Carregamento eficiente do dataset (otimizar dtypes para 565 MB)
- [x] Análise de missing values e estratégia de tratamento
- [x] Estatísticas descritivas completas
- [x] Distribuição da variável target (`ARRIVAL_DELAY`)
- [x] Análise por dimensão:
  - [x] Por companhia aérea (top/bottom 5)
  - [x] Por aeroporto de origem (heatmap geográfico)
  - [x] Por dia da semana e mês
  - [x] Por horário do dia (morning/afternoon/evening/night)
- [x] Heatmap de correlação entre features numéricas
- [x] Análise de causas de atraso (breakdown dos 5 tipos)
- [x] Documentar 5+ insights relevantes

### Critérios de Conclusão

- Dataset entendido, sem surpresas nos dados
- Hipóteses levantadas para os modelos
- Figuras salvas em `outputs/figures/eda_*.png`

---

## Fase 2 — Modelagem Supervisionada (3-4 dias)

**Notebook:** `notebooks/02_supervisionado.ipynb`  
**Comando:** `/train`

### Tarefas

- [x] Feature engineering (IS_DELAYED, DEPARTURE_HOUR, DEPARTURE_PERIOD, SEASON, IS_WEEKEND, ROUTE, DISTANCE_BUCKET)
- [x] Pipeline de pré-processamento (One-Hot AIRLINE, Target Enc. Origin/Dest/Route, Ordinal Period/Season/Bucket, split 80/20 estratificado)
- [x] **Classificação:**
  - [x] Logistic Regression (baseline) — métricas + confusion matrix
  - [x] Random Forest — métricas + feature importance
  - [x] XGBoost — métricas + curva ROC + early stopping
  - [x] Tabela comparativa de Accuracy, Precision, Recall, F1, AUC (salva em outputs/)
  - [x] Cross-validation (StratifiedKFold, k=5, amostra 500k estratificada)
  - [x] Precision-Recall Curve (desbalanceamento 17.91%)
- [x] **Regressão:**
  - [x] Linear Regression (baseline) — RMSE, MAE, R²
  - [x] XGBoost Regressor — RMSE, MAE, R² + early stopping
  - [x] Análise de resíduos + distribuição de resíduos
- [x] Modelos salvos em `outputs/models/` (clf_logreg, clf_rf, clf_xgb, reg_linear, reg_xgb, scaler)

### Critérios de Conclusão

- Pelo menos 2 modelos comparados para classificação e 2 para regressão
- Melhor modelo identificado com justificativa
- Feature importance documentada

---

## Fase 3 — Modelagem Não Supervisionada (2-3 dias)

**Notebook:** `notebooks/03_nao_supervisionado.ipynb`  
**Comando:** `/cluster`

### Tarefas

- [x] Agregação de 10 features por aeroporto (avg_delay, taxi_out, distance, cancel_rate, 4 causas de delay)
- [x] Elbow method para determinar k ótimo (k=2..10)
- [x] Silhouette score para validar k ótimo
- [x] KMeans com k ótimo — heatmap Z-score + interpretação dinâmica por cluster
- [x] PCA (2 componentes) — scatter colorido por cluster + labels dos 20 maiores hubs
- [x] Loadings do PCA — contribuição de cada feature aos componentes
- [x] Mapa interativo Folium (HTML) + mapa estático matplotlib
- [x] Heatmap das features médias por cluster (Z-score)
- [x] Exportar `outputs/airport_clusters.csv`
- [x] DBSCAN para detecção de aeroportos anômalos (opcional concluído)
- [x] Clustering de companhias aéreas por perfil (opcional concluído)

### Critérios de Conclusão

- Clusters com interpretação de negócio clara
- Visualizações que comunicam os grupos encontrados

---

## Fase 4 — Resultados e Apresentação (1-2 dias)

**Notebook:** `notebooks/04_resultados.ipynb`  
**Comando:** `/report`

### Tarefas

- [x] Resumo executivo do projeto
- [x] Responder as 4 perguntas guia com evidência dos dados
- [x] Tabela comparativa final de modelos
- [x] Discussão de limitações (dataset 2015, features ausentes, etc.)
- [x] Próximos passos propostos
- [x] Revisar todos os notebooks para narrativa coesa

### Critérios de Conclusão

- Todos os requisitos da especificação atendidos
- Notebooks executáveis do início ao fim sem erros
- Conclusões claras e críticas

---

## Extras Opcionais (se houver tempo)

- [ ] Dashboard interativo com Streamlit
- [ ] Mapa de rotas com atrasos no Folium
- [ ] Detecção de anomalias com Isolation Forest
- [ ] Análise de feriados (dataset de feriados EUA 2015)
- [ ] DBSCAN como alternativa ao KMeans

---

## Métricas de Sucesso

| Critério | Meta |
|----------|------|
| EDA com insights documentados | ≥ 5 insights |
| Modelos de classificação comparados | ≥ 2 |
| Modelos de regressão comparados | ≥ 2 |
| Abordagem não supervisionada | ≥ 1 (KMeans + PCA) |
| Notebooks executáveis sem erro | 100% |
| Apresentação com limitações | Obrigatório |

---

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Dataset de 565 MB lento de carregar | Usar `dtype` otimizado; sample para prototipar |
| Overfitting nos modelos | Cross-validation + regularização |
| Desbalanceamento de classes | Class weights ou SMOTE |
| Memória insuficiente | Chunks com `pd.read_csv(chunksize=...)` |
