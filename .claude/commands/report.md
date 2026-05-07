# /report — Apresentação de Resultados

Execute a Fase 4 do TechChallenger3: consolidação e apresentação crítica dos resultados.

## Pré-requisito

Fases 1, 2 e 3 concluídas.

## O que fazer

### Estrutura do Notebook Final (`04_resultados.ipynb`)

#### 1. Resumo Executivo (2-3 parágrafos)
- Contexto do problema e objetivo
- Abordagem metodológica
- Principal resultado

#### 2. Achados da EDA
- Top 5 insights com gráficos de suporte
- Responder as 4 perguntas guia do projeto:
  1. Quais aeroportos são mais críticos?
  2. Que características aumentam a chance de atraso?
  3. Atrasos por dia/horário?
  4. Agrupamento de aeroportos?

#### 3. Resultados de Modelagem Supervisionada
- Tabela comparativa: Logistic Regression vs Random Forest vs XGBoost
- Melhor modelo: justificativa da escolha
- Feature importance: quais variáveis mais influenciam
- Limitações: viés de dados, overfitting, dataset balanceado?

#### 4. Resultados de Clustering
- Descrição dos clusters encontrados
- Mapa geográfico dos clusters
- Business insight: o que cada cluster representa na prática?

#### 5. Limitações e Crítica
- Dataset de 2015 — sazonalidade pode ter mudado
- Variáveis externas ausentes: clima real, ocupação do voo, manutenção
- Desbalanceamento de classes (maioria sem atraso)
- Aeroportos pequenos com poucos voos distorcem métricas

#### 6. Próximos Passos
- Adicionar dados de clima em tempo real
- Testar modelos de série temporal (LSTM, Prophet)
- Criar API de predição em tempo real
- Dashboard interativo com Streamlit/Dash

### Exportar Apresentação

Criar `outputs/apresentacao_final.pdf` ou slides com os principais gráficos.

## Output esperado

- Notebook `notebooks/04_resultados.ipynb` completo e narrativo
- Todas as figuras referenciadas existem em `outputs/figures/`
- Conclusões claras e alinhadas com a especificação do projeto
