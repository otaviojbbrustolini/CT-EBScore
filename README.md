# CT-EBScore — Cancer-Testis Evidence Score com Dirichlet–Multinomial Empirical Bayes

> **Status: método exploratório (prévia).** O `ct_score_eb` resume o padrão de classificações na literatura, **não** a probabilidade de um gene ser biologicamente Cancer-Testis (CT). Verificações preditivas revelaram inadequação importante para contagens positivas de `yes`.

## Visão geral

**CT-EBScore** (*Cancer-Testis Empirical Bayes Score*) é um score contínuo de evidência literária para priorização de genes candidatos a Cancer-Testis. O script `ct_score_dm.R` ajusta um modelo **Dirichlet–Multinomial com Empirical Bayes (DM-EB)** às contagens por gene nas categorias `yes`, `no` e `unknown`. A coluna central da saída chama-se **`ct_score_eb`**.

A regularização Bayesiana (*shrinkage*) evita que um gene com apenas `1 yes / 0 no` receba automaticamente o mesmo valor máximo da proporção bruta que outro gene com dezenas de evidências positivas. Genes com `yes = 0`, `no = 0` ou nenhuma classificação conclusiva são preservados.

As categorias representam:

- **`yes`:** o artigo fornece evidência favorável à classificação CT do gene;
- **`no`:** o artigo fornece evidência contrária à classificação CT;
- **`unknown`:** o artigo não oferece evidência conclusiva; **não equivale a `no`**.

## Dados de entrada

Arquivo CSV/TSV, com uma linha por gene:

| Campo | Significado |
|---|---|
| `geneID` | Identificador único do gene |
| `yes` | Número de classificações favoráveis |
| `no` | Número de classificações contrárias |
| `unknown` | Número de classificações inconclusivas |
| `total` | Soma das três categorias |

Exemplo (`ct_score_example.csv`):

```csv
geneID,yes,no,unknown,total
RGS6,0,59,311,370
RNF17,4,2,102,108
CTAG1A,1061,29,3056,4146
SPANXC,15,0,70,85
BAGE3,1,0,20,21
```

O script verifica se os valores são inteiros não negativos, `total = yes + no + unknown`, `total >= 1` e `geneID` não é vazio nem duplicado. Cada classificação artigo–gene deve pertencer a apenas uma categoria.

**Pressupostos:** as classificações são tratadas como condicionalmente independentes, dadas as probabilidades latentes de cada gene, e todos os genes compartilham uma distribuição Dirichlet a priori. Artigos correlacionados, classificações repetidas e diferenças sistemáticas de cobertura bibliográfica podem violar essas hipóteses.

## Metodologia estatística

### 1. Modelo de três categorias

Para cada gene $g$ definimos:

$$
\mathbf{x}_g=(Y_g,N_g,U_g), \qquad n_g=Y_g+N_g+U_g.
$$

Assumimos uma distribuição multinomial condicional:

$$
\mathbf{x}_g\mid\boldsymbol{\theta}_g
\sim\operatorname{Multinomial}(n_g,\boldsymbol{\theta}_g),
$$

em que $\boldsymbol{\theta}_g=(\theta_{Y,g},\theta_{N,g},\theta_{U,g})$ é o vetor de probabilidades latentes para `yes`, `no` e `unknown`, com soma igual a 1. O prior compartilhado é:

$$
\boldsymbol{\theta}_g\sim
\operatorname{Dirichlet}(\alpha_Y,\alpha_N,\alpha_U).
$$

Ao integrar $\boldsymbol{\theta}_g$, obtemos a distribuição marginal Dirichlet–Multinomial. Os hiperparâmetros positivos $\boldsymbol{\alpha}$ são estimados **por máxima verossimilhança marginal**, usando os genes disponíveis:

$$
\ell(\boldsymbol{\alpha})=\sum_g\left[
\log\Gamma(\alpha_0)-\log\Gamma(n_g+\alpha_0)
+\sum_{k\in\{Y,N,U\}}
\left(\log\Gamma(x_{gk}+\alpha_k)-\log\Gamma(\alpha_k)\right)
\right]+\mathrm{constante},
$$

onde $\alpha_0=\alpha_Y+\alpha_N+\alpha_U$. O termo combinatório multinomial é omitido na otimização por não depender de $\boldsymbol{\alpha}$. O script utiliza `optim()` do R com `L-BFGS-B`, transformação logarítmica dos parâmetros e múltiplas inicializações. No procedimento Empirical Bayes, os hiperparâmetros estimados são tratados como fixos na etapa posterior.

### 2. Distribuição posterior por gene

Pela conjugação Dirichlet–Multinomial:

$$
\boldsymbol{\theta}_g\mid\mathbf{x}_g\sim
\operatorname{Dirichlet}
(Y_g+\alpha_Y,N_g+\alpha_N,U_g+\alpha_U).
$$

A média posterior para a categoria $k$ é:

$$
E(\theta_{k,g}\mid\mathbf{x}_g)=
\frac{x_{gk}+\alpha_k}{n_g+\alpha_0},
\qquad k\in\{Y,N,U\}.
$$

### 3. Definição do `ct_score_eb`

Como `unknown` indica **ausência de conclusão**, separamos a proporção de evidências *conclusivas* da predominância positiva entre elas. Definimos:

$$
Q_g=\frac{\theta_{Y,g}}{\theta_{Y,g}+\theta_{N,g}}.
$$

Pela propriedade de agregação da Dirichlet:

$$
Q_g\mid\mathbf{x}_g\sim
\operatorname{Beta}(Y_g+\alpha_Y,N_g+\alpha_N).
$$

Portanto, o **CT-EBScore** é a média posterior de $Q_g$:

$$
\boxed{
\operatorname{CT\text{-}EBScore}_g
=E(Q_g\mid\mathbf{x}_g)
=\frac{Y_g+\alpha_Y}
{Y_g+N_g+\alpha_Y+\alpha_N}
}
$$

O resultado pertence ao intervalo [0, 1] e representa a **fração esperada de classificações `yes` entre as classificações conclusivas**, sob o modelo. Ele **não** é a probabilidade de o gene ser biologicamente CT. A categoria `unknown` influencia a estimação conjunta do prior e as medidas de conclusividade, mas não entra diretamente no denominador do CT-score condicionado a `yes`/`no`.

Genes sem classificações conclusivas ($Y_g+N_g=0$) recebem a média a priori $\alpha_Y/(\alpha_Y+\alpha_N)$; são identificados como `no_conclusive_evidence` e **não devem ser interpretados como evidência específica do gene**.

### 4. Probabilidades e intervalos

Denotando $a_g=Y_g+\alpha_Y$ e $b_g=N_g+\alpha_N$, a distribuição Beta posterior fornece:

$$
P(Q_g>t\mid\mathbf{x}_g)=1-F_{\operatorname{Beta}(a_g,b_g)}(t),
\qquad t\in\{0.5,0.8\}.
$$

São as colunas `p_ct_gt_05` e `p_ct_gt_08`. Os limites `ct_low95` e `ct_high95` correspondem aos quantis 2,5% e 97,5% da Beta posterior. A proporção posterior de classificações conclusivas é separada:

$$
E(\theta_{Y,g}+\theta_{N,g}\mid\mathbf{x}_g)
=\frac{Y_g+N_g+\alpha_Y+\alpha_N}{n_g+\alpha_0}.
$$

Essa quantidade é exportada em `p_conclusive_eb`.

## Estudo piloto: 7.117 genes

A base exploratória contém **7.117 genes**, dos quais **6.150 (86,41%)** possuem `yes = 0`; **386 (5,42%)** não possuem classificações conclusivas; a mediana da proporção `unknown/total` foi **94,44%**.

O ajuste Empirical Bayes produziu:

| Parâmetro | Valor |
|---|---:|
| $\alpha_Y$ | 0,05097322 |
| $\alpha_N$ | 1,59607962 |
| $\alpha_U$ | 19,84443989 |
| $\kappa=\alpha_0$ | 21,49149 |

O vetor de médias do prior foi $(0,002371786;\ 0,074265647;\ 0,923362567)$. A média *conclusiva* a priori, $\alpha_Y/(\alpha_Y+\alpha_N)$, é aproximadamente **0,03095**, valor que **não deve ser confundido com prevalência de genes CT**.

Para esse prior específico, a fórmula torna-se:

$$
\operatorname{CT\text{-}EBScore}_g=
\frac{Y_g+0.05097322}{Y_g+N_g+1.64705284}.
$$

### Exemplos de score

| Gene | `yes` | `no` | `yes/(yes+no)` | `ct_score_eb` |
|---|---:|---:|---:|---:|
| CTAG1A | 1061 | 29 | 0,9734 | 0,9720 |
| SPANXC | 15 | 0 | 1,0000 | 0,9041 |
| BAGE3 | 1 | 0 | 1,0000 | 0,3970 |
| RNF17 | 4 | 2 | 0,6667 | 0,5297 |
| RGS6 | 0 | 59 | 0,0000 | 0,0008 |

Os exemplos ilustram o *shrinkage*: dois genes com proporção bruta igual a 1 podem receber scores distintos conforme o volume de evidência conclusiva. **Esses exemplos não constituem validação biológica.**

## Execução

**Requisito:** instalação do R; o script usa apenas funções do R base.

```bash
Rscript ct_score_dm.R input.csv ct_scores.csv
```

Para reutilizar o prior do estudo piloto e comparar resultados no mesmo referencial:

```bash
Rscript ct_score_dm.R input.csv ct_scores.csv \
  --alpha=0.05097322,1.59607962,19.84443989
```

São aceitos arquivos separados por vírgula, tabulação ou ponto e vírgula, com detecção automática. Para forçar o separador:

```bash
Rscript ct_score_dm.R input.tsv ct_scores.csv --sep=tab
Rscript ct_score_dm.R --help
```

O script gera `ct_scores.csv` e `ct_scores_parameters.csv` (nome base da saída seguido por `_parameters.csv`). Quando os parâmetros são fornecidos por `--alpha`, não ocorre otimização e os campos de convergência/verossimilhança do arquivo de parâmetros permanecem sem valores.

### Colunas adicionais de saída

| Coluna | Interpretação |
|---|---|
| **`ct_score_eb`** | Média posterior de $Q_g$; score exploratório principal |
| `p_ct_gt_05` | Probabilidade posterior de $Q_g>0,5$ |
| `p_ct_gt_08` | Probabilidade posterior de $Q_g>0,8$ |
| `ct_low95`, `ct_high95` | Intervalo de credibilidade de 95% para $Q_g$ |
| `p_yes_eb`, `p_no_eb`, `p_unknown_eb` | Médias posteriores das três categorias |
| `p_conclusive_eb` | Média posterior da proporção de classificações conclusivas |
| `n_conclusive` | Quantidade de classificações conclusivas (`yes + no`) |
| `evidence_status` | `conclusive_evidence` ou `no_conclusive_evidence` |
| `rank_ct_score` | Ranking decrescente de `ct_score_eb` (empates compartilham posição) |
| `rank_p_ct_gt_08` | Ranking decrescente de `p_ct_gt_08` (empates compartilham posição) |

**Uso recomendado na prévia:** observar `ct_score_eb` em conjunto com `n_conclusive`, `p_ct_gt_08`, intervalo de credibilidade e `evidence_status`. Não adotar limiares arbitrários como classificação biológica validada.

**Reprodutibilidade:** recalcular o prior em outro conjunto de genes pode alterar os scores. Para comparações entre tabelas, documentar e fixar o prior com `--alpha`. Isso não resolve automaticamente problemas de calibração nem diferenças na composição dos datasets.

## Diagnósticos e limitações conhecidas

### Verificação preditiva

Realizamos **300 simulações preditivas paramétricas**, mantendo o número total de classificações observado em cada gene e simulando contagens sob os hiperparâmetros ajustados:

| Estatística | Observado | Média simulada | Intervalo preditivo 95% |
|---|---:|---:|---:|
| Genes com `yes = 0` | 6.150 | 6.202,31 | 6.153,48–6.252,53 |
| Genes com `yes >= 10` | 56 | 239,97 | 213,48–268,00 |
| Média de `yes` | 0,574 | 3,399 | 2,234–5,482 |
| Percentil 99 de `yes` | 6 | 45,63 | 35,92–59,60 |
| Máximo de `yes` | 1.061 | 4.146,80 | 953,98–14.165,03 |

O modelo representa aproximadamente a frequência de zeros, **mas superestima intensamente as contagens positivas moderadas/altas**. Portanto, **o ajuste global ainda não é satisfatório**. Esse procedimento é uma verificação preditiva *plug-in* com hiperparâmetros fixados, não uma verificação preditiva plenamente Bayesiana.

Um Beta–Binomial Empirical Bayes condicionado às classificações `yes`/`no` também superestimou o número de genes com `yes >= 10` (cerca de **181 previstos**, contra **56 observados**).

### Limitações principais

1. **Inadequação preditiva:** probabilidades de excedência e intervalos podem estar mal calibrados sob o prior ajustado.
2. **Evidência textual versus estado biológico:** o score mede o padrão de classificações de artigos, e não a verdade biológica da classificação CT.
3. **Dependência entre artigos/classificações:** observações relacionadas podem superestimar o tamanho amostral efetivo e tornar os intervalos excessivamente estreitos.
4. **Viés de seleção e cobertura bibliográfica:** a proporção agregada de `yes` varia muito conforme o volume de literatura por gene, contrariando uma média latente compartilhada simples.
5. **Incerteza dos hiperparâmetros:** a estimação Empirical Bayes do prior não é propagada nos intervalos por gene.
6. **Dados incompletos:** alterações na coleta e classificação dos papers podem modificar os parâmetros e o ranking.

### Próximas avaliações

- Análise de sensibilidade do prior e da estabilidade das posições dos 50–100 primeiros genes.
- Verificações preditivas estratificadas por `total` e por `n_conclusive`.
- Comparação com regressão Beta–Binomial e modelos de mistura, incluindo avaliação preditiva em dados reservados.
- Validação com genes CT e não CT independentemente curados, medindo AUROC/AUPRC e qualidade dos limiares de decisão.
- Caso o objetivo passe a ser uma probabilidade *biológica* calibrada de CT, desenvolvimento de um modelo supervisionado adicional com rótulos biológicos independentes.

## Aviso de uso

> **Caution: exploratory CT evidence scores; model calibration should be checked.**

O **CT-EBScore** está disponível para **prévia de ranking e priorização de revisão científica**. `ct_score_eb` e `p_ct_gt_08` **não devem ser apresentados como probabilidades validadas de o gene ser Cancer-Testis** enquanto as limitações do modelo e a validação biológica não forem resolvidas.
