# /eda — Executar Análise Exploratória de Dados

Execute a Fase 1 do TechChallenger3: análise exploratória completa do dataset de voos.

## O que fazer

### Seção 1 — Carregamento e Inspeção Inicial

- Carregar `dados/flights.csv`, `dados/airports.csv`, `dados/airlines.csv`
- Otimizar dtypes para reduzir memória:
  - `int16`: YEAR, MONTH, DAY, DAY_OF_WEEK, FLIGHT_NUMBER
  - `float32`: todas as colunas de delay e tempo
  - `category`: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, CANCELLATION_REASON
- Reportar: shape, uso de memória antes/depois, dtypes finais
- Fazer merge com `airlines.csv` para ter nome completo da companhia

---

### Seção 2 — Qualidade dos Dados

- `df.describe()` para todas as colunas numéricas
- Contar valores únicos por coluna categórica (AIRLINE, ORIGIN, DEST, TAIL_NUMBER)
- Mapa de missing values:
  - Barplot de % de NaN por coluna
  - Atenção: delay columns NaN = voo sem atraso → preencher com 0
  - DEPARTURE_TIME / ARRIVAL_TIME NaN = voo cancelado → separar em subset
  - CANCELLATION_REASON NaN = voo não cancelado → OK

---

### Seção 3 — Análise da Variável Target

- Distribuição de `ARRIVAL_DELAY`: histograma com escala linear e log
- Histograma de `DEPARTURE_DELAY` lado a lado com `ARRIVAL_DELAY`
- **Análise de recuperação de tempo no ar:**
  - Criar coluna `TIME_RECOVERED = DEPARTURE_DELAY - ARRIVAL_DELAY`
  - Voos que partem atrasados conseguem recuperar tempo? (boxplot por airline)
- Outliers extremos:
  - Quantos voos com `ARRIVAL_DELAY > 180 min` (3h)?
  - Quantos com `ARRIVAL_DELAY > 360 min` (6h)?
  - Listar os 10 maiores atrasos do dataset com contexto (airline, rota, data)
- Taxa de atraso por threshold: `> 0 min`, `> 15 min`, `> 60 min`, `> 120 min`

---

### Seção 4 — Análise por Companhia Aérea

- Top/bottom 5 airlines por **taxa de atraso** (ARRIVAL_DELAY > 15 min)
- **Pontualidade operacional:** comparar `ELAPSED_TIME` vs `SCHEDULED_TIME` por airline
  - Airlines que consistentemente terminam mais rápido/lento do que o previsto
- **Causa dominante de atraso por airline:**
  - Heatmap: airlines × tipo de delay (AIR_SYSTEM, WEATHER, AIRLINE, LATE_AIRCRAFT, SECURITY)
  - Identifica quais airlines têm problemas internos vs. externos
- Taxa de cancelamento por airline (barplot ordenado)
- Taxa de desvio (`DIVERTED`) por airline

---

### Seção 5 — Análise Temporal

- Atrasos médios por `MONTH` — linha do tempo (sazonalidade)
- Atrasos médios por `DAY_OF_WEEK` — barplot (1=Seg … 7=Dom)
- **Análise por hora do dia:**
  - Converter `SCHEDULED_DEPARTURE` (HHMM) → hora inteira (0–23)
  - Criar `DEPARTURE_HOUR = SCHEDULED_DEPARTURE // 100`
  - Barplot de atraso médio por hora — identifica "rush hours" de atraso
  - Distribuição de voos ao longo do dia (volume × horário)
- Heatmap 2D: `DAY_OF_WEEK` × `MONTH` com cor = atraso médio

---

### Seção 6 — Análise por Aeroporto

- Top 10 aeroportos de **origem** com maior atraso médio (mín. 1000 voos)
- Top 10 aeroportos de **destino** com maior atraso médio
- Comparar: os mesmos aeroportos são problemáticos como origem e destino?
- **Análise de taxiamento:**
  - Top 10 aeroportos com maior `TAXI_OUT` médio (congestionamento no solo)
  - Correlação entre `TAXI_OUT` e `DEPARTURE_DELAY` (scatter com linha de tendência)
  - Scatter `TAXI_IN` vs `ARRIVAL_DELAY`
- Volume de voos por aeroporto — distribuição (poucos hubs dominam?)

---

### Seção 7 — Análise de Rotas e Distância

- Distribuição de `DISTANCE` por histograma — voos curtos, médios, longos
- Criar categoria `DISTANCE_BUCKET`: curto (<500mi), médio (500–1500mi), longo (>1500mi)
- Atraso médio por `DISTANCE_BUCKET` — voos curtos atrasam mais?
- Top 20 rotas mais voadas (ORIGIN → DESTINATION)
- Top 20 rotas com maior atraso médio (mín. 100 voos na rota)

---

### Seção 8 — Cancelamentos e Desvios

- Taxa geral de cancelamento e desvio no dataset
- Distribuição de `CANCELLATION_REASON`:
  - A = Airline, B = Weather, C = NAS (air traffic), D = Security
  - Pizza chart + contagem absoluta
- Cancelamentos por mês (há sazonalidade?)
- Cancelamentos por airline (quais têm mais?)
- Cancelamentos por aeroporto de origem (top 10)

---

### Seção 9 — Correlações e Features para Modelagem

- Heatmap de correlação entre todas as features numéricas relevantes:
  - DEPARTURE_DELAY, ARRIVAL_DELAY, TAXI_OUT, TAXI_IN, AIR_TIME, ELAPSED_TIME, SCHEDULED_TIME, DISTANCE
  - Delay types: AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY, WEATHER_DELAY
- Scatter matrix (pairplot) das 5 features mais correlacionadas com ARRIVAL_DELAY
- **Aeronaves problemáticas (TAIL_NUMBER):**
  - Top 10 aeronaves com maior atraso médio (mín. 50 voos)
  - Isso sugere uso de TAIL_NUMBER como feature ou que manutenção influencia atrasos

---

### Seção 10 — Insights e Hipóteses

Célula Markdown final com:
- Mínimo 8 insights numerados, cada um com evidência (número ou gráfico de referência)
- Para cada insight: levantar hipótese para uso na modelagem supervisionada
- Exemplo de estrutura:
  ```
  **Insight 1:** Voos com TAXI_OUT > 30 min têm DEPARTURE_DELAY médio 3x maior (18 vs 6 min).
  → Hipótese: TAXI_OUT será feature importante no modelo de classificação.
  ```

## Output esperado

- Script `notebooks/01_eda.py` completo e executável (padrão `# %%`)
- Figuras salvas em `outputs/figures/eda_*.png` (uma por seção)
- Mínimo 8 insights documentados com hipóteses para modelagem

## Padrao para script Python Analiticos 

Todo script Python de analise de dados deve usar delimitadores de celula `# %%`, tornando-o executavel celula por celula no VS Code Python Interactive Window (igual a um notebook, mas em .py).

* `# %%` -- inicia uma celula de codigo. O VS Code exibe "Run Cell" acima de cada uma.
* `# %% [markdown]` -- celula de texto/markdown (titulos, descricoes de secao)

O arquivo continua funcionando normalmente como script via python arquivo.py


## O dicionario dos dados flights.csv


| Coluna | Descrição | Tipo / Unidade |
|---|---|---|
| YEAR | Ano do voo (ex.: 2015) | Inteiro |
| MONTH | Mês do voo (1 a 12) | Inteiro |
| DAY | Dia do mês do voo (1 a 31) | Inteiro |
| DAY_OF_WEEK | Dia da semana (1 = Segunda, 7 = Domingo) | Inteiro |
| AIRLINE | Código da companhia aérea (ex.: AA = American Airlines) | Categórica |
| FLIGHT_NUMBER | Número do voo | Inteiro |
| TAIL_NUMBER | Número de registro da aeronave | Texto |
| ORIGIN_AIRPORT | Código IATA do aeroporto de origem (ex.: ATL) | Categórica |
| DESTINATION_AIRPORT | Código IATA do aeroporto de destino | Categórica |
| SCHEDULED_DEPARTURE | Horário de partida programado (HHMM) | Inteiro |
| DEPARTURE_TIME | Horário real de partida (HHMM) | Inteiro |
| DEPARTURE_DELAY | Atraso na partida (em minutos) | Numérico |
| TAXI_OUT | Tempo gasto taxiando até a decolagem (em minutos) | Numérico |
| WHEELS_OFF | Horário em que o avião decolou (HHMM) | Inteiro |
| SCHEDULED_TIME | Tempo total programado de voo (em minutos) | Numérico |
| ELAPSED_TIME | Tempo total real de voo (em minutos) | Numérico |
| AIR_TIME | Tempo no ar (em minutos) | Numérico |
| DISTANCE | Distância entre origem e destino (em milhas) | Numérico |
| WHEELS_ON | Horário em que as rodas tocaram o solo (HHMM) | Inteiro |
| TAXI_IN | Tempo taxiando até o portão de desembarque (em minutos) | Numérico |
| SCHEDULED_ARRIVAL | Horário de chegada programado (HHMM) | Inteiro |
| ARRIVAL_TIME | Horário de chegada real (HHMM) | Inteiro |
| ARRIVAL_DELAY | Atraso na chegada (em minutos) | Numérico |
| DIVERTED | Indica se o voo foi desviado (1 = sim, 0 = não) | Binária |
| CANCELLED | Indica se o voo foi cancelado (1 = sim, 0 = não) | Binária |
| CANCELLATION_REASON | Motivo do cancelamento (A = Airline, B = Weather, C = NAS, D = Security) | Categórica |
| AIR_SYSTEM_DELAY | Atraso causado por controle de tráfego aéreo | Numérico |
| SECURITY_DELAY | Atraso causado por problemas de segurança | Numérico |
| AIRLINE_DELAY | Atraso causado pela companhia aérea | Numérico |
| LATE_AIRCRAFT_DELAY | Atraso causado por chegada tardia da aeronave | Numérico |
| WEATHER_DELAY | Atraso causado por condições meteorológicas | Numérico |