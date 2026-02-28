## Full project flow for 5 weeks of experiments + 1 to 2 weeks for report/poster

### 0) Final scope and decisions

* **Goal:** build a **risk-adjusted hallucination detector** that predicts whether an answer is **unsupported by the available evidence**, then **abstains** when risk is high.
* **Main transfer question:** does a detector trained on one evidence regime transfer to another **for the same base model**?
* **Evidence regimes:**

  * **PHANTOM**: evidence is given in prompt
  * **HalluLens retrieval subset**: evidence is retrieved from Wikipedia and attached
* **Prediction target:**

  * **y = 1** if the answer is **not supported by the provided evidence**
  * **y = 0** if the answer **is supported by the provided evidence**
* **Detector model:**

  * **Primary**: logistic regression
  * **Secondary ablation**: tiny MLP
* **Primary operating point:** optimize for **safety first**, but always report full **risk coverage curves**
* **Cross-model transfer:** **out of scope**
* **Expandable feature design:** keep the detector input as a feature table so we can later add more hallucination indicators from OpenAI-style hallucination work without changing the pipeline

### Why this is the right scope

* It stays close to the slide
* It uses existing tools instead of building models from scratch
* It is defensible in 5 weeks
* It leaves room for one or two small ablations without exploding scope

---

## A) Choose components

### 1) Base model and tools

**Base model**

* Pick **one lightweight open LLM** as the main generator
* Optional second open model only if time remains after main results

**Recommended libraries**

* **transformers** for model loading and generation
* **datasets** for dataset loading and preprocessing
* **torch** for inference and simple NN experiments
* **accelerate** for easier multi-GPU or efficient inference if needed
* **scikit-learn** for logistic regression, metrics, calibration baselines
* **wandb** or **tensorboard** for experiment tracking
* **pandas / numpy** for tables and feature storage
* **matplotlib / seaborn** only for plots in report or poster
* **sentence-transformers** for semantic clustering if needed
* **evaluate** for standard metric helpers
* **requests** for MediaWiki API retrieval
* **gradio** only at the end for a small demo if time allows

**Do**

* Use Hugging Face models and model cards
* Use Hugging Face datasets and dataset cards
* Use PapersWithCode only to choose a reasonable NLI model and not to expand scope
* Track every run with fixed configs

**Do not**

* Do not train a new LLM
* Do not build retrieval, NLI, or calibration from scratch
* Do not add too many datasets or too many models

---

## B) Data preparation

### 2) Build the two evaluation regimes

#### Regime 1: PHANTOM

Use PHANTOM as-is.

* Input = **evidence chunk + question**
* Label meaning = whether the answer is supported by the given chunk

#### Regime 2: HalluLens retrieval-grounded subset

Construct a subset from HalluLens.

For each chosen HalluLens question:

1. Extract keywords from the **question only**
2. Query **MediaWiki API**
3. Retrieve **top-k passages** from Wikipedia, with **k = 3** as the default
4. Store retrieved evidence with the question

**Why top-k and not one snippet**

* It reduces false unsupported cases caused by missed evidence
* It keeps the task fairer and more defensible

**Libraries**

* **requests**
* **wikipedia-api** or direct **MediaWiki API**
* **datasets** for storing processed subset
* **pandas** for bookkeeping

### 3) Create labels for the HalluLens retrieval subset

Use **LLM-as-judge** for labels.

**Judge input**

* question
* retrieved evidence
* served answer

**Judge rubric**

* **Supported**
* **Contradicted**
* **NEI** = not enough information

**Binary mapping**

* Supported → **y = 0**
* Contradicted or NEI → **y = 1**

**Important rule**

* Supported is allowed **only if the judge cites exact evidence spans**

### 4) Leakage control

Use different tools for labels and groundedness feature.

* **Labels y**: LLM-as-judge with fixed rubric and fixed prompt
* **Feature x4**: NLI-based evidence groundedness score

**Leakage note to use in presentation/report**
To reduce circularity, label generation and evidence-groundedness scoring are separated: labels are assigned by a fixed LLM judge with span citations, while the groundedness feature is computed independently using NLI.

### 5) Create splits

For each regime:

* train
* validation
* test

Use fixed random seeds and save split IDs.

**Simple explanation**
This makes sure we train, tune, and test on separate data, and do not accidentally use test examples early.

---

## C) Generation phase

### 6) Generate one served answer and K sampled answers

For each example:

**PHANTOM prompt**

* evidence chunk + question

**HalluLens retrieval subset prompt**

* retrieved wiki evidence + question

Generate:

1. **Served answer**
2. **K sampled answers**

**Recommended K**

* Start with **K = 5**
* If affordable, test **K = 3, 5, 8** in a small sensitivity study

**Generation settings**

* Keep prompt template fixed
* Keep decoding settings fixed
Decoding settings are the controls for **how the LLM generates text** after we give it a prompt.

They matter because features depend on generation behavior, especially:

* **served answer**
* **K sampled answers**
* uncertainty
* disagreement
* semantic entropy

decoding settings fixed: use the **same generation controls** across runs so results are fair and comparable.

## The main decoding settings we’ll use

### 1) Temperature

Controls how random the next token choice is.

* **low temperature** like 0.2 to 0.5 → more deterministic, safer, less diverse
* **higher temperature** like 0.8 to 1.0 → more diverse, more random

For the project:

* for the **served answer**, use a **low temperature** or greedy decoding
* for the **K sampled answers**, use a **non-zero temperature** so we actually get variation

### 2) Top-p

Also called nucleus sampling.

The model only samples from the smallest set of tokens whose total probability adds up to **p**.

Example:

* **top_p = 0.9** means sample only from the most likely tokens covering 90% of the probability mass

For the project:

* useful for sampled answers
* common setting: **top_p = 0.9 or 0.95**

### 3) Top-k

The model samples only from the top **k** most likely next tokens.

Example:

* **top_k = 50** means only sample from the 50 most likely tokens

In practice:

* many people use **top-p** and leave top-k alone
* we do **not** need to overcomplicate this

### 4) do_sample

This tells the model whether to sample at all.

* **do_sample=False** → deterministic decoding, usually greedy
* **do_sample=True** → stochastic decoding

For project:

* **served answer**: usually `do_sample=False`
* **K sampled answers**: `do_sample=True`

### 5) Max new tokens

Maximum length of generated answer.

This prevents very long outputs and keeps answers comparable.

For project:

* set one fixed limit, for example **64**, **96**, or **128** depending on dataset answer length

### 6) Repetition penalty / no-repeat n-gram

These help reduce weird repetition.

we probably do **not** need to make this a major variable.
Just keep it fixed if we use it.

---

## What we should likely do

### Served answer

Use a stable setting like:

* `do_sample=False`
* low temperature or greedy
* fixed `max_new_tokens`

This makes the main answer consistent.

### K sampled answers

Use:

* `do_sample=True`
* `temperature=0.7`
* `top_p=0.9`
* fixed `max_new_tokens`

This gives enough diversity for:

* self-consistency disagreement
* semantic entropy

---

## Why this matters for the project

If we change decoding settings too much, we features change for reasons unrelated to hallucination.

For example:

* higher temperature can artificially increase disagreement
* longer outputs can create more unsupported claims
* different top-p settings can change semantic entropy

So we want to avoid this problem by saying:

> We fix decoding settings across all experiments and only vary them in a small sensitivity analysis if needed.

---

## Simple way to think about it

Decoding settings are just the model’s **answer style controls**:

* how random
* how long
* how exploratory
* how repeat-heavy

we keep them fixed so the detector is learning hallucination risk, not just differences caused by generation randomness.

## A good default for the project

**Served answer**

* `do_sample=False`
* `max_new_tokens=96`

**K sampled answers**

* `do_sample=True`
* `temperature=0.7`
* `top_p=0.9`
* `max_new_tokens=96`

Decoding settings are the generation controls like temperature, top-p, and answer length. We keep them fixed so uncertainty features are comparable across examples.

* Log temperature, top-p, max tokens, seed

**Libraries**

* **transformers**
* **torch**
* **accelerate**
* **wandb**

**Simple explanation**
The served answer is what the model would normally say. The sampled answers help measure uncertainty and disagreement.

---

## D) Feature generation

### 7) Compute the 4 core features

For each example, create one feature vector:

[
x = [x_1, x_2, x_3, x_4]
]

#### Feature 1: token uncertainty

* Average token NLL or token entropy of the served answer

**Library**

* **transformers** with logits
* **torch**

#### Feature 2: self-consistency disagreement

* Compare the K sampled answers and score how much they disagree

**Practical method**

* Use pairwise semantic similarity or contradiction proxy
* Keep it simple and consistent

**Library**

* **sentence-transformers**
* **scikit-learn**
* optionally simple lexical overlap baseline

#### Feature 3: semantic entropy

* Cluster sampled answers by meaning
* Compute entropy over clusters

**Library**

* **sentence-transformers** for embeddings
* **scikit-learn** for clustering
* **scipy / numpy** for entropy

#### Feature 4: evidence groundedness

* Run NLI between the answer claims and the evidence
* Aggregate entailment / contradiction / neutral into one score

**Simple aggregation**

* split answer into sentences
* run NLI for each sentence against evidence
* compute:

  * percent entailed
  * percent contradiction
  * percent neutral
* convert to a single groundedness score

**Library**

* **transformers** for NLI model
* **nltk** or **spacy** for sentence splitting

### 8) Keep feature design expandable

Do not hard-code the detector to exactly 4 features forever.

Instead:

* save all features in a **feature table**
* define a clear feature registry
* start with the 4 core features
* later, add extra hallucination indicators if useful

**Examples of future expandable features**

* answer length
* number of claims
* citation style / hedging indicators
* contradiction count
* prompt-position sensitivity
* extra indicators inspired by OpenAI hallucination work

**Important**
These extra features are optional and only added if they improve results clearly. The 4-feature setup remains the main experiment.

---

## E) Detector training

### 9) Train the detector over features

**Primary detector**

* logistic regression

**Why**

* simple
* interpretable
* stable for 4 to a few more features
* easy to defend in presentation

**Secondary ablation**

* tiny MLP with 1 hidden layer of 8 to 16 units

**Libraries**

* **scikit-learn** for logistic regression
* **torch** or **sklearn MLPClassifier** for tiny MLP

**Simple explanation**
The detector learns how to combine uncertainty and groundedness signals into one hallucination risk score.

### 10) Calibration

Fit calibration on the **source validation split** only.

**Primary**

* temperature scaling

**Optional comparison if time allows**

* isotonic regression

**Libraries**

* custom temperature scaling in **torch**
* **scikit-learn** for isotonic regression

**Metrics**

* ECE
* Brier score

**Simple explanation**
Calibration turns the detector score into a probability we can actually trust.

---

## F) Abstention policy

### 11) Choose the abstention threshold

Rule:

* **Answer** if calibrated risk ≤ threshold
* **Abstain** if calibrated risk > threshold

**Primary selection strategy**

* choose threshold on **source validation**
* prioritize **safety**
* still show full **risk coverage** and **accuracy coverage** curves

**What to say if asked safety vs utility**
We use safety-first threshold selection for the headline result, but we report full risk coverage curves so utility tradeoffs are visible.

**Libraries**

* **numpy**
* **scikit-learn**
* **matplotlib**

---

## G) Transfer evaluation

### 12) Main experiment: cross-regime transfer within one model

#### Direction 1

* train detector on **PHANTOM train**
* calibrate and choose threshold on **PHANTOM val**
* freeze detector, calibration, and threshold
* test on **HalluLens retrieval test**

#### Direction 2

* train detector on **HalluLens retrieval train**
* calibrate and choose threshold on **HalluLens retrieval val**
* freeze everything
* test on **PHANTOM test**

**Report**

* AUROC
* AUPRC
* ECE
* Brier
* risk coverage
* accuracy coverage

**Simple explanation**
we train on one evidence setup, then test whether the same detector still works on the other one without retuning.

12.1) Transfer under shift

Then paste a tightened version like this:

12.1) Transfer under shift

After the main frozen transfer experiment, run one small domain adaptation comparison.

We evaluate two target-regime settings:

Setting A: strict transfer

detector, calibration, and abstention threshold are frozen from source validation

Setting B: light adaptation

detector weights are frozen

only calibration is refreshed on a small target validation split

optionally reselect the abstention threshold on that same target validation split

Important constraints

do not retrain the detector

do not redesign features

do not change the generator or prompt setup

Why this is included

it keeps the project small

it directly tests whether transfer failure is mainly due to calibration drift under regime shift

it gives a stronger answer to the domain adaptation limitation without expanding scope too much

What this tells us

This comparison helps distinguish:

detector failure under shift

calibration failure under shift

One-line summary for presentation/report

Our primary result is strict frozen cross-regime transfer. To address distribution shift, we also test a lightweight adaptation setting where detector weights are frozen and only target-side calibration is updated.

---

## H) Baselines

### 13) Compare against simple baselines

Use the same evaluation protocol for all baselines.

* no abstention
* token uncertainty only
* self-consistency only
* semantic entropy only
* evidence groundedness only
* random abstention

**Why**
This shows whether combining signals is actually better than using just one signal.

---

## I) Robustness and failure analysis

### 14) Small, realistic robustness checks

Keep this small and focused.

#### Check 1: K sensitivity

* compare K = 3, 5, 8 on a subset

#### Check 2: prompt/decoding sensitivity

* run one small alternate prompt or temperature setting

#### Check 3: retrieval miss analysis

* manually inspect a sample of NEI cases
* tag retrieval failures vs true unsupported answers

#### Check 4: confident wrong cases

* find examples with low uncertainty but y = 1
* show whether groundedness helps catch them

#### Check 5: evidence placement in PHANTOM

* if possible, test a small subset with different evidence placement or longer context

**Do not**

* do not turn this into a big cross-model benchmark
* do not add many extra datasets

---

## J) Addressing the audience questions directly

### 15) Final answers to the open questions from earlier notes

#### Q1. Logistic regression or small NN?

**Decision:** logistic regression is the main detector, tiny MLP is a small ablation.

#### Q2. Safety or utility?

**Decision:** safety-first as headline, full curves reported.

#### Q3. What exactly is hallucination here?

**Answer:** evidence-unsupportedness, not general world-factuality.

#### Q4. What is the prediction unit?

**Answer:** compute groundedness at the **sentence level**, aggregate to an **answer-level** risk score.

#### Q5. Are labels reliable?

**Answer:** yes, but we still do a manual audit sample and report judge limitations.

#### Q6. Is there leakage?

**Answer:** minimized by separating LLM judge for labels from NLI for groundedness feature.

#### Q7. What transfer is being tested?

**Answer:** cross-regime transfer across evidence setups and domains, for a fixed base model.

#### Q8. Are the features universal?

**Answer:** that is the hypothesis, but we support it with ablations and failure analysis, not by overclaiming.

#### Q9. Does it generalize across models?

**Answer:** out of scope. This project is within-model transfer only.

#### Q10. What about prompt and temperature changes?

**Answer:** we run one small sensitivity check and keep configs fixed otherwise.

#### Q11. Is this just a hardness detector?

**Answer:** partially related, so we control for length and inspect confident-wrong cases where hardness alone does not explain failure.

#### Q12. Are we just using another model to solve the task?

**Answer:** only partially. NLI gives one groundedness feature, but the final detector combines several independent signals.

#### Q13. What about confident wrong answers?

**Answer:** this is a main failure mode we explicitly analyze. It is exactly why groundedness is included in addition to entropy features.

---

## K) Main limitations and how to address them

### 16) Limitations section to build into the report from the start

#### 1. Limited dataset diversity

* Only PHANTOM and HalluLens retrieval subset are used
* **Address:** state clearly that conclusions are for these two regimes and grounded QA settings

#### 2. Retrieval recall problems

* A correct answer may look unsupported because retrieved evidence missed the fact
* **Address:** use top-k retrieval and report retrieval misses separately

#### 3. Label noise from LLM judge

* Judge can be imperfect
* **Address:** fixed rubric, span citation requirement, small manual audit, fixed judge across runs

#### 4. Possible circularity

* If labeler and feature scorer are too similar, results can inflate
* **Address:** use LLM judge for labels and NLI for x4

#### 5. Calibration brittleness under shift

Source-calibrated probabilities may degrade on target regime

Address: explicitly measure ECE and Brier after transfer, and add a lightweight adaptation comparison where detector weights stay frozen but calibration and possibly threshold are refreshed on a small target validation split

#### 6. K-sample compute cost

* More samples improve uncertainty features but increase runtime
* **Address:** run a small K sensitivity study and justify final K

#### 7. Feature set may miss some hallucination types

* 4 features may not capture everything
* **Address:** keep feature design expandable and report this as future work

#### 8. Within-model only

* Results may not transfer to other generators
* **Address:** state scope honestly and avoid overclaiming

#### 9. NLI errors

* NLI may fail on long evidence or subtle claims
* **Address:** keep evidence short, split answers into sentences, inspect failure cases

#### 10. Threshold depends on risk preference

* A single threshold may not fit all stakeholders
* **Address:** report full curves, not just one threshold

---

## L) 5-week execution plan for a 4-person team

## Week 1: lock scope, data, prompts, and tools

### What to do

* choose the main open LLM
* choose the NLI model
* load PHANTOM and HalluLens
* implement MediaWiki retrieval
* define the fixed prompt template
* define the judge rubric
* define split strategy
* create repo structure and experiment tracker

### Libraries

* transformers
* datasets
* torch
* requests
* wandb or tensorboard
* pandas

### Team split

* **Person 1:** dataset loading and split pipeline
* **Person 2:** Wikipedia retrieval and HalluLens subset creation
* **Person 3:** generation pipeline and prompt config
* **Person 4:** LLM judge rubric, logging format, audit sheet

### Deliverables

* one fixed project config file
* processed HalluLens retrieval subset
* written label rubric
* dry run on 20 examples

---

## Week 2: generate answers and create labels

### What to do

* run served-answer generation on both regimes
* run K sampled generations
* run LLM judge on HalluLens retrieval subset
* save all raw outputs in structured files
* manually audit 30 to 50 judged examples

### Libraries

* transformers
* torch
* accelerate
* wandb
* pandas / json

### Team split

* **Person 1:** PHANTOM generations
* **Person 2:** HalluLens retrieval generations
* **Person 3:** sampled-answer generation and storage
* **Person 4:** LLM judge execution and manual audit

### Deliverables

* all served answers
* K sampled answers
* labeled HalluLens retrieval subset
* short audit memo with common judge mistakes

---

## Week 3: compute features and build feature table

### What to do

* compute x1 token uncertainty
* compute x2 disagreement
* compute x3 semantic entropy
* compute x4 NLI groundedness
* merge into one feature table
* verify no leakage between labels and features

### Libraries

* transformers
* sentence-transformers
* scikit-learn
* scipy
* nltk or spacy
* pandas / numpy

### Team split

* **Person 1:** x1 token uncertainty
* **Person 2:** x2 disagreement
* **Person 3:** x3 semantic entropy
* **Person 4:** x4 NLI groundedness and sentence splitting
* all together: merge and validate feature table

### Deliverables

* one final CSV or parquet per regime with x and y
* basic descriptive plots
* feature sanity-check notebook

---

## Week 4: train detector, calibrate, and run baselines

### What to do

* train logistic regression
* run tiny MLP ablation
* fit temperature scaling
* run baseline abstention methods
* generate source-regime metrics
* choose source validation threshold

### Libraries

* scikit-learn
* torch
* numpy
* matplotlib
* wandb

### Team split

* **Person 1:** logistic regression + evaluation
* **Person 2:** tiny MLP ablation
* **Person 3:** calibration and threshold selection
* **Person 4:** baseline runs and plotting

### Deliverables

* within-regime detector results
* calibration results
* baseline comparison plots

---

## Week 5: transfer, failure analysis, and final figures

### What to do

* run PHANTOM → HalluLens transfer
* run HalluLens → PHANTOM transfer
* compute AUROC, AUPRC, ECE, Brier, risk coverage
* do failure analysis on retrieval misses, confident wrong, and long-context issues
* finalize tables and figures

### Libraries

* scikit-learn
* pandas
* numpy
* matplotlib
* wandb

### Team split

* **Person 1:** transfer direction 1
* **Person 2:** transfer direction 2
* **Person 3:** failure analysis and example collection
* **Person 4:** final plots, result tables, and slide-ready visuals

### Deliverables

* final experimental results
* final result tables
* 3 to 5 strong qualitative examples
* final conclusion on whether transfer worked

---

## Week 6: report writing

### What to do

* write method section from the finalized pipeline
* write experiment setup with exact configs
* write results and limitations
* write discussion carefully without overclaiming
* add appendix with judge rubric and prompts

### Team split

* **Person 1:** intro + related work
* **Person 2:** methods
* **Person 3:** experiments + results
* **Person 4:** discussion + limitations + appendix
* all edit together for consistency

### Deliverables

* full report draft
* clean figure captions
* appendix with prompts, models, splits, and feature definitions

---

## Week 7: poster and polish

### What to do

* turn report into poster
* keep poster focused on task, method, main results, failure cases
* add one simple pipeline figure
* add one main risk coverage figure
* optional Gradio demo if time remains

### Libraries / tools

* PowerPoint, Google Slides, or Figma for poster
* Gradio for demo if needed

### Team split

* **Person 1:** poster lawet
* **Person 2:** method figure
* **Person 3:** results figures
* **Person 4:** demo or appendix material

### Deliverables

* final poster
* final presentation backup slides
* optional demo

---

## M) What to say in the report and poster

### Core contribution statement

We formulate risk-adjusted hallucination detection as supervised prediction of evidence-unsupportedness from uncertainty and groundedness signals, then test whether the detector transfers across evidence regimes for a fixed open-source LLM.

### One-line method statement

We combine token uncertainty, self-consistency disagreement, semantic entropy, and NLI-based groundedness in a lightweight detector, calibrate its output, and abstain when predicted hallucination risk is high.

### Honest scope statement

This project studies cross-regime transfer within a fixed generator model, not cross-model generalization.

---	

## Bottom line plan
* Keep the project to **one main model, two regimes, four+ core features**
* Use **LLM-as-judge for labels** and **NLI for groundedness feature**
* Use **logistic regression/NN as the main detector**
* Show **transfer, calibration, and abstention**
* Keep features **expandable** for extra hallucination indicators later
* Use existing libraries throughout
* Do a few small robustness checks, not many
* Spend the last 1 to 2 weeks on a strong report and poster, not on adding extra experiments