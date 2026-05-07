# /cluster — Modelagem Não Supervisionada

Execute a Fase 3 do TechChallenger3: clustering de aeroportos e redução de dimensionalidade.

## Pré-requisito

Fase 1 (EDA) concluída. Dataset limpo disponível.

## O que fazer

### Preparação do Perfil de Aeroportos

Agregar `flights.csv` por `ORIGIN_AIRPORT` para criar um perfil por aeroporto:
- `avg_departure_delay`: média de DEPARTURE_DELAY
- `avg_arrival_delay`: média de ARRIVAL_DELAY
- `cancellation_rate`: % de voos cancelados
- `total_flights`: volume total de voos
- `weather_delay_rate`: % do atraso causado por clima
- `airline_delay_rate`: % do atraso por culpa da companhia
- `avg_distance`: distância média das rotas

Fazer merge com `airports.csv` para ter CITY, STATE, LAT, LON.

### KMeans Clustering

1. Normalizar features com StandardScaler
2. Elbow method: testar k de 2 a 10, plotar inércia
3. Silhouette Score: confirmar k ótimo
4. Treinar KMeans com k ótimo (sugestão: k=4 ou 5)
5. Interpretar cada cluster:
   - Cluster X: "Aeroportos pequenos, baixo atraso"
   - Cluster Y: "Hubs grandes, alto atraso por sistema"

### PCA para Visualização

1. Aplicar PCA(n_components=2) nas features normalizadas
2. Plotar scatter 2D colorido por cluster KMeans
3. Plotar loadings do PCA para entender contribuição de cada feature
4. Verificar variância explicada pelos 2 componentes

### Visualizações

- Salvar em `outputs/figures/cluster_*.png`
- Mapa geográfico dos aeroportos coloridos por cluster (usando LAT/LON)
- Heatmap das features médias por cluster
- Scatter PCA com labels dos aeroportos mais relevantes

### Análise Adicional (opcional)

- DBSCAN para detectar aeroportos anômalos
- Clustering de companhias aéreas por perfil de atraso

## Output esperado

- Notebook `notebooks/03_nao_supervisionado.ipynb` completo
- DataFrame com aeroportos e seus clusters em `outputs/airport_clusters.csv`
- Interpretação textual de cada cluster


## Padrao para script Python Analiticos 

Todo script Python de analise de dados deve usar delimitadores de celula `# %%`, tornando-o executavel celula por celula no VS Code Python Interactive Window (igual a um notebook, mas em .py).

* `# %%` -- inicia uma celula de codigo. O VS Code exibe "Run Cell" acima de cada uma.
* `# %% [markdown]` -- celula de texto/markdown (titulos, descricoes de secao)

O arquivo continua funcionando normalmente como script via python arquivo.py
