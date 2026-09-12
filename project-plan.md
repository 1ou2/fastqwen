# Project Plan — "J'ai fait parler mon modèle 10x plus vite"

> Vidéo YouTube : la course d'inférence. Rendre Qwen3.8-27B confortable en local
> sur DGX Spark via quantization, distillation d'un draft model maison et
> speculative decoding — le tout mesuré, étape par étape, dans vLLM.
>
> Répertoire projet : `/home/gabriel/dev/fastqwen`
> Date de démarrage : 22/08/2026
>
> **Ce document est le plan de référence pour l'implémentation.** Tout écart
> constaté en cours de route se documente dans `benchmarks/results.md`, pas ici.

---

## 1. Objectif du projet

Vidéo YouTube originale sur les techniques d'optimisation d'inférence des LLMs.

- **Narration** : Qwen3.8-27B est excellent mais trop lent pour tourner
  confortablement en local. On l'optimise étape par étape, avec un benchmark
  visuel à chaque chapitre.
- **Métrique du titre** : deux chiffres distincts, jamais confondus —
  1. **tokens/s en décodage** (batch=1, longueur de génération fixée,
     `reasoning_effort` figé) : la mesure de vitesse pure ;
  2. **temps total jusqu'à la réponse finale complète** à `reasoning_effort`
     identique : c'est ce que l'audience ressent, et c'est ce chiffre qui
     portera le titre. Toute variation du nombre de tokens de raisonnement
     est explicitement mentionnée à l'écran, sinon le gain est faux.
- **Contraintes** : évaluations systématiquement sur DGX Spark ; H100 (cluster
  de 4) uniquement pour l'entraînement ; max 3 jours consécutifs de GPU.
- **Périmètre** : texte uniquement. L'encodeur vision du 27B n'est ni quantizé
  ni évalué.

### 1.1 Objectifs d'apprentissage (critères de réussite personnels)

Le projet est autant un support pédagogique qu'une vidéo. **vLLM est le moteur
unique du projet**, précisément pour monter en compétence dessus : chaque
mécanisme ci-dessous doit être compris, manipulé, et expliqué dans la vidéo.

| Mécanisme | Où il est travaillé |
|---|---|
| PagedAttention, gestion du KV cache, mémoire unifiée | Étape 0, étape D |
| Continuous batching, `--max-num-seqs`, saturation | Étape 0, étape D |
| Chunked prefill, TTFT, prefix caching | Étape D |
| CUDA graphs, `--enforce-eager`, déterminisme | Protocole (§6) |
| Quantization hors ligne (NVFP4, FP8) et kernels | Étape A |
| KV cache quantizé | Étape D |
| Speculative decoding : vérification, taux d'acceptation, γ | Étape C |
| Multi-token prediction (MTP) natif | Étape C |
| Distillation SFT et KD sur logits | Étape B |
| Profiling : métriques Prometheus, `vllm bench` | Protocole (§6) |

---

## 2. Matériel

| Machine | Usage |
|---|---|
| NVIDIA DGX Spark (GB10, sm_121, mémoire unifiée 128 Go, ~273 Go/s) | Tous les benchmarks et inférences (c'est la machine de l'audience) |
| 4× H100 | Génération du dataset, calibration, entraînement du draft model |

---

## 3. Modèles retenus

| Rôle | Modèle | Statut |
|---|---|---|
| Target (héros de la vidéo) | **Qwen3.8-27B** | ✅ support day-0 (vLLM, SGLang, llama.cpp), déjà installé |
| Draft « gratuit » intégré | **Têtes MTP natives du 3.8-27B** | ✅ multi-token prediction embarqué (cf. §3.3) |
| Élève / draft model | **Qwen3.5-2B initialisé puis distillé** | ✅ compatibilité tokenizer vérifiée |

### 3.1 Caractéristiques du target qui contraignent tout le reste

- **27B dense** (pas de MoE) : chaque paramètre est lu à chaque token → le
  décodage est purement **memory-bound** (§5).
- **Pile d'attention hybride** : 64 couches alternant 3 blocs *Gated DeltaNet*
  pour 1 bloc *gated-attention*. Conséquence directe : le KV cache est bien
  plus petit qu'un transformer classique (seule une couche sur quatre en a un),
  ce qui renforce la thèse « le goulot est la lecture des poids ».
- **Contexte natif 262 144 tokens** (~1M avec YaRN) : la pression mémoire vient
  surtout des poids et du prefill, pas du cache.
- **Thinking hybride, activé par défaut**, piloté par `reasoning_effort`
  (`low` / `medium` / `xhigh`). Il n'existe plus de variante « Instruct ».
- **Vision** : encodeur multimodal présent, hors périmètre.
- **Licence Apache 2.0**.
- **La famille 3.8 n'a aucun modèle sous 27B** : il n'existe pas de petit frère
  natif à utiliser comme draft. C'est la justification de l'étape B.

### 3.2 Pourquoi Qwen3.5-2B comme élève

Le 3.8 n'existe qu'en 27B (et au-dessus). Le plus petit modèle de la même
lignée de tokenizer est dans la génération précédente : **Qwen3.5-2B**. Il sert
de point d'initialisation, puis il est distillé depuis le 27B lui-même.

### 3.3 Les têtes MTP : le draft intégré

Le 27B embarque des têtes de **multi-token prediction (MTP)** : le modèle
prédit nativement plusieurs tokens en avant, ce qui donne un speculative
decoding « gratuit » (aucun entraînement, aucun GPU H100 consommé). Les stacks
communautaires sur GB10 tournent d'ailleurs en `NVFP4 + MTP3`.

Le MTP est donc un **bras obligatoire de l'étape C**, pas un bonus. Il fournit
la ligne de comparaison honnête : si le draft distillé maison ne bat pas le MTP
natif, c'est le résultat de la vidéo, et il est intéressant à expliquer.

### 3.4 Compatibilité des tokenizers (vérifiée le 22/08/2026)

Script : `check_tokenizer_compat.py` (racine du repo). Ce script a servi à
**valider la faisabilité** du choix de l'élève ; il doit être durci (§3.5)
avant de servir de garde-fou reproductible dans le pipeline.

Résultats :

- `vocab.json` et `merges.txt` bit à bit identiques ; cœur du tokenizer identique
- Vocabulaire de base : 248 044 tokens des deux côtés
- Encodage fonctionnel strictement identique (tests FR, guillemets « », emoji,
  multilingue, tokens chat)
- Seules différences : 7 tokens spéciaux audio/TTS côté 3.8 (IDs 248070-248076,
  **au-delà** du vocabulaire de base, donc appendés et sans décalage d'ID) et le
  `chat_template`

**Décision** : compatibles. Initialiser l'élève depuis Qwen3.5-2B tel quel, sans
extension de vocabulaire. Utiliser le **chat template du 3.8 partout** là où un
template est requis (génération du dataset, inférence finale, évaluations
conversationnelles) — et **jamais** pour le calcul de perplexité (§6, point 4).

### 3.5 Contrat de compatibilité à verrouiller

Le speculative decoding exige une correspondance **exacte** token ↔ ID. Avant
l'étape B, `check_tokenizer_compat.py` doit affirmer, sur des **révisions HF
pinnées**, les exigences dures suivantes :

1. Égalité de l'ensemble des paires (token → ID) sur tout le vocabulaire partagé.
2. IDs identiques pour `bos`, `eos`, `pad` et les tokens de chat
   (`im_start`, `im_end`, et tout spécial utilisé par le template du 3.8).
3. Tout token supplémentaire du 3.8 est strictement au-delà de la taille du
   vocabulaire de base : aucun ID partagé n'est décalé.
4. `apply_chat_template(...)` avec le template du 3.8 rend des IDs identiques
   sur les deux tokenizers.
5. Un fichier de tokenizer absent est un **échec**, jamais un avertissement.

Différences **tolérées** et listées explicitement : `chat_template`,
`clean_up_tokenization_spaces`, tokens audio/TTS. Le verdict du script ne doit
pas dépendre de ces différences cosmétiques.

À confirmer en conditions réelles : vLLM valide la compatibilité des tokenizers
au chargement d'un `draft_model` ; noter le message exact en cas d'échec.

---

## 4. Déroulé par étapes (entrée → sortie)

```
Qwen3.8-27B (BF16, thinking hybride, têtes MTP natives)
  │
  ├──> A: Quantization ─────────────────────────┐
  │     FP8 → NVFP4 4-bit → palier agressif     │
  │                                             ├──> C: Speculative decoding
  ├──> B: Distillation du draft ────────────────┤     MTP natif · draft brut
  │     (prof = 3.8-27B lui-même, sur H100)     │     draft distillé · MTP+draft
  │                                             │
  └──> D: KV cache & serving ───────────────────┘
        (prefill, cache, CUDA graphs)
                    │
                    └──> E: Climax vidéo
```

### Étape 0 — Prérequis & baseline

- **Entrée** : Qwen3.8-27B BF16 sur DGX Spark
- **Prérequis à valider avant toute mesure** (bloquants) :
  1. **Image vLLM fonctionnelle sur GB10 (sm_121)** : vLLM standard ne
     s'installe pas tel quel sur Spark ; utiliser une image communautaire
     **pinnée par digest** (référence : `eugr/spark-vllm-b12x`) ou construire
     depuis `eugr/spark-vllm-docker`. Noter digest + version vLLM exacte : c'est
     le socle de toute la reproductibilité.
  2. **Support FP8** sur GB10 confirmé par un run court.
  3. **Support NVFP4** (chargement d'un checkpoint 4-bit + génération) confirmé.
  4. **Mode MTP** : identifier la clé `--speculative-config` exacte supportée par
     cette version de vLLM et vérifier un run court.
- **Action** : benchmark complet avec le protocole uniforme (§6), avec
  `reasoning_effort` **figé et documenté**.
- **Sortie** : chiffres de référence — tout sera comparé à ça.

**Choix du runtime d'inférence : vLLM partout, pour tout.** Un seul moteur pour
la baseline, les quantizations, les benchmarks et le speculative decoding.
Motif double : (1) sinon on mesure le runtime et pas l'optimisation ;
(2) **vLLM est le moteur que je veux maîtriser** — c'est un objectif
d'apprentissage du projet, pas un choix par défaut. Concrètement, on utilise
`vllm bench latency|throughput|serve` et l'endpoint Prometheus pour les mesures,
avec un wrapper mince (`scripts/bench_spark.py`) plutôt qu'un banc maison.

### Étape A — Quantization

Trois paliers, du plus sûr au plus agressif.

1. **FP8** : le plus simple. Qwen publie des dépôts `-FP8` pour tous ses modèles
   ouverts : on part du checkpoint officiel, sans conversion. Quasi sans perte.
2. **NVFP4 4-bit** : le palier principal, et le **format natif de Blackwell**
   (FP4 avec échelle FP8 par micro-blocs). C'est le chemin matériel du GB10 :
   c'est donc là que se trouve le vrai gain de vitesse. Produit hors ligne avec
   NVIDIA ModelOpt / `llm-compressor`, puis chargé par vLLM (format modelopt
   détecté depuis la config du checkpoint).
3. **La limite** : méthode réellement plus agressive (3-bit / 2-bit type HQQ,
   ou recette mixte dégradée) pour montrer la casse. Si la dégradation n'est pas
   spectaculaire, c'est aussi un résultat à raconter (« les méthodes modernes
   sont robustes jusqu'à 4-bit »).

- **Sortie** : le même 27B compressé, mesuré aux 3 paliers
- **Gain attendu** : ~3x vitesse décode, ~3,4x mémoire (56 Go → ~16 Go)
- **Insight** : la quantization optimise la lecture mémoire, pas le calcul — le
  modèle reste un dense qui génère token par token
- **Point de vigilance** : les blocs *Gated DeltaNet* et l'encodeur vision ne
  sont pas des `Linear` standard. Si un palier ne se charge pas ou casse la
  qualité, la recette de repli est une **recette mixte** : garder ces couches
  (et le ViT) en BF16/FP8, ne quantizer que les projections d'attention et les
  MLP. Ce repli est un résultat pédagogique, pas un échec.

**Calibration NVFP4** (et tout palier qui en demande une) :

- **Dataset mixte obligatoire**, ~512 séquences de 2048 tokens, répartition :
  - texte français généraliste (Wikipedia FR, presse, prose) — le gros du volume
  - **code** (Python, shell, JSON, diffs) — protège les capacités de génération
    de code
  - **tool-calling / agentique** : trajectoires d'appels d'outils, schémas
    JSON, refus quand aucun outil ne convient — protège exactement les
    capacités que le protocole mesure (§6, point 5)
  - une part de conversations avec traces de raisonnement, puisque le modèle
    raisonne par défaut
- **Règle de contamination** : aucun prompt du jeu d'évaluation
  (`benchmarks/`) n'apparaît dans la calibration. Vérification automatique.
- **Traçabilité** : hash du set de calibration consigné dans le manifeste du run.

### Étape B — Distillation : création du draft model *(sur H100)*

Le Qwen3.8 n'existe qu'en 27B : il n'y a pas de petit modèle natif compatible.
On fabrique donc soi-même le draft model manquant.

- **B1. Génération du dataset** (le prof est le Qwen3.8-27B lui-même)

  Le dataset est construit en **tranches**, parce que le taux d'acceptation
  dépend de la distribution : le draft doit être bon là où la vidéo fait la
  démonstration.

  | Tranche | Contenu | Volume cible |
  |---|---|---|
  | S1 | Conversationnel général FR + EN (OpenHermes-2.5, UltraChat) | ~120M tokens |
  | S2 | **Code** (instructions de génération, correction, explication) | ~25M tokens |
  | S3 | **Tool-calling / agentique** (schémas d'outils, appels chaînés, refus) | ~15M tokens |
  | S4 | Traces de raisonnement (optionnel, selon budget) | +30-60M tokens |

  - Génération par vLLM sur 4×H100, avec le chat template du 3.8
  - `reasoning_effort` **figé** pour la génération aussi : c'est ce qui rend le
    budget de tokens prévisible (le thinking actif par défaut multiplierait le
    volume par 2 à 4 et ferait exploser le créneau H100)
  - Durée : ~14-24 h sur les 4×H100
  - **Jeu de validation hors ligne** : ~2 000 prompts représentatifs des
    tranches S2/S3 + des prompts de la démo finale, réservés et **jamais vus à
    l'entraînement** — ils servent à prédire le taux d'acceptation sur H100
    avant de consommer du temps Spark (§6, point 7)

- **B2. SFT de l'élève**
  - Entrée : Qwen3.5-2B initialisé tel quel + dataset B1
  - Sortie : **draft model aligné statistiquement avec le 27B**
  - Durée : ~6-10 h sur 1×H100
  - **Option KD sur logits** : +5 h, à déclencher seulement si la porte G2
    (§7) n'est pas franchie

### Étape C — Speculative decoding : la convergence

Quatre bras, mesurés avec le même protocole :

- **C1. MTP natif seul** — les têtes MTP du 27B, γ ∈ {1,3} (référence : les
  stacks communautaires utilisent `MTP3`). C'est le bras « gratuit ».
- **C2. Draft Qwen3.5-2B non distillé** — résultat volontairement montré :
  gains décevants, taux d'acceptation faible. C'est le contrôle qui **prouve**
  l'utilité de l'étape B.
- **C3. Draft distillé** (issu de l'étape B) — le cœur de la vidéo.
- **C4. MTP + draft distillé** — si la version de vLLM le permet, sinon
  documenté comme limite de l'outil.

**Protocole de γ** : γ n'est pas deviné sur Spark. Le gain théorique du
speculative decoding vaut

```
       1 - α^(γ+1)
S  =  ─────────────────        α = taux d'acceptation, c = coût du draft
       (1 - α) · (γ·c + 1)         exprimé en fraction du coût du target
```

`α` et `c` sont mesurés **hors ligne sur H100** (`scripts/offline_acceptance.py`,
balayage γ ∈ {1,3,5,7}), et seuls les 2 meilleurs candidats sont validés sur
Spark. Le taux d'acceptation est mesuré **séparément sur les segments de
raisonnement et sur la réponse finale** : c'est là qu'un élève 2B décroche, et
c'est une mesure inédite et intéressante à montrer.

**Vérification d'exactitude (obligatoire, et très visuelle)** : en greedy, le
speculative decoding avec vérification est **exact**. On valide donc C1-C4 par
un **test d'identité token-à-token** face au target seul, sur les 30 prompts
qualitatifs. C'est la preuve que l'accélération est gratuite en qualité — et
cela remplace une campagne qualité complète, qui serait du temps perdu.

- **Sortie** : qualité du 27B, vitesse multipliée, γ optimal, coût du draft
- **Insight clé** : le speculative decoding n'accélère que si le draft prédit
  comme le target — d'où la nécessité de la distillation
- **Bonus Spark** : le décode batch=1 y est memory-bound, conditions idéales :
  vérifier γ+1 tokens coûte ≈ une seule lecture des poids du target

### Étape D — KV cache & serving *(la marche qui manque au débit)*

Le speculative decoding attaque le décodage. L'autre moitié du problème est le
prefill et la gestion du cache. C'est aussi le terrain de jeu vLLM du projet.

Optimisations à tester une par une, en **A/B à variable unique**, sur la
configuration gagnante de l'étape C :

1. **FP8 KV cache** (`--kv-cache-dtype fp8`) — à noter : avec l'attention
   hybride, seule une couche sur quatre a un KV cache ; mesurer le gain réel
   est en soi un résultat.
2. **Chunked prefill** (`--max-num-batched-tokens`) — effet direct sur le TTFT
   en contexte long.
3. **Prefix caching** — décisif pour les prompts système / outils répétés.
4. **CUDA graphs** et `--enforce-eager` — arbitrage vitesse / déterminisme.
5. **Continuous batching** : `--max-num-seqs` croissant, pour tracer la courbe
   débit vs latence (le Spark est mono-utilisateur, mais la courbe est
   pédagogique).

**Règle** : toute variante qui ne tient pas dans le bruit (§6.1) est rejetée et
documentée comme telle. Une optimisation rejetée est un résultat publiable.

- **Sortie** : la configuration de serving finale, et la décomposition
  « ce qui vient de la quantization / du draft / du serving »

### Étape E — Climax vidéo

Course finale sur Spark : baseline BF16 vs pipeline complet (27B NVFP4 + draft
distillé + config de serving de l'étape D), même question, chronomètre à
l'écran. Tableau cumulé des gains étape par étape.

- Même `reasoning_effort` des deux côtés, et **nombre de tokens de raisonnement
  affiché** à côté du chrono
- **Contrôle « pourquoi pas un modèle plus petit ? »** : une ligne de référence
  avec un modèle plus petit servi à débit comparable (famille 3.5/3.6, MoE à
  quelques milliards d'actifs). C'est la première objection de l'audience et
  elle se ferme pour ~1 h de Spark.

---

## 5. Le décor : la physique du problème

Segment « la physique » de la vidéo, et garde-fou contre les promesses
exagérées. Le Spark a ~273 Go/s de bande passante mémoire unifiée :

| Configuration | Poids lus / token | Plafond théorique batch=1 |
|---|---|---|
| BF16 | ~56 Go | **≈ 5 tokens/s** |
| 4-bit (NVFP4) | ~16 Go | **≈ 17 tokens/s** |
| 4-bit + speculative decoding | ~16 Go, γ+1 tokens vérifiés | **≈ 17 × tokens acceptés par passe** |

Le speculative decoding ne dépasse pas ce plafond : il augmente le nombre de
tokens produits **par lecture des poids**. Avec α = 0,7 et γ = 3, on accepte en
moyenne ~2,5 tokens par passe de target → plafond ≈ 40 tokens/s. Les chiffres
communautaires sur GB10 (19-40 tokens/s selon le type de contenu) sont cohérents
avec ce calcul.

**Conséquence narrative** : le « 10x » du titre ne peut pas venir du seul
tokens/s. Il vient du temps total jusqu'à la réponse, et il faut être explicite
sur la part qui vient de la vitesse et la part qui vient du nombre de tokens.

---

## 6. Protocole d'évaluation (uniforme à chaque étape)

Sur DGX Spark, mêmes prompts et paramètres à chaque mesure, tout passe par
vLLM. Mesures via `vllm bench` + wrapper `scripts/bench_spark.py` + métriques
Prometheus de vLLM.

1. **Vitesse** : tokens/s en décodage (batch=1), time-to-first-token (TTFT),
   et — pour la vidéo — temps total jusqu'à la réponse complète.
   **3 runs par mesure** (variance thermique du Spark), médiane et p95.
2. **Mémoire** : mémoire unifiée peak, échantillonnée **hors vLLM**
   (`tegrastats`), pas seulement le chiffre rapporté par le moteur.
3. **Énergie** : puissance moyenne (W) pendant le run. Gratuit à collecter,
   très parlant en vidéo.
4. **Qualité — perplexité** ⭐ : held-out Wikipédia FR (~200k tokens). Métrique
   la plus sensible aux dégradations fines de quantization :
   FP8 ≈ +0-2% PPL · 4-bit ≈ +2-5% · méthodes agressives ≈ +20%+.
   - **Calcul sur texte brut** : ni chat template, ni raisonnement. Fenêtres
     glissantes de 512 tokens, pour éviter les effets de contexte et de cache.
   - Procédure documentée dans le script, pas dans un notebook.
5. **Qualité — capacités générales** : `benchmarks/quality/quality-200.jsonl`,
   **en complément** des autres niveaux :
   - 180 items auto-gradés, grading déterministe : gsm8k (80, exact match
     numérique), humaneval (40, exécution de code + tests), ifeval (40, suivi
     strict d'instructions), agentic_coding (20, exécution)
   - ⚠️ Les 20 items `hard_reasoning` (grading `manual`) sont **exclus du
     score** (non reproductibles) ; réservés à d'éventuelles démonstrations
     qualitatives
   - **Prérequis avant usage** : (a) les 20 items `agentic_coding` n'ont pas de
     tests — leur champ `reference` est un placeholder (`"exec+asserts"`) qui
     laisse le modèle écrire ses propres asserts ; ils doivent être **réécrits
     avec de vrais tests externes** avant de compter dans le score, sinon on
     mesure de l'auto-notation ; (b) toute exécution de code généré passe par un
     **sandbox** (conteneur sans réseau, timeouts, FS jetable)
   - **Extraction des réponses** : le raisonnement est retiré avant de grader
     (le thinking est actif par défaut) ; l'extracteur est testé sur des sorties
     réelles, pas supposé
   - Résolution statistique : avec 180 items, l'erreur-type est de ±3,7 pts et
     une comparaison **non appariée** ne détecte rien sous ~7-8 pts. Les écarts
     FP8/BF16 attendus sont dans ce bruit → **la comparaison se fait item par
     item, appariée (McNemar)**, pas sur les totaux. La perplexité reste la
     métrique fine.
   - Contenu en anglais : normal, les dégradations sont corrélées entre langues ;
     le FR reste couvert par la perplexité wiki-FR et les prompts qualitatifs
6. **Qualité — agentique** ⭐ (Qwen3.8 excelle dans ce domaine, il faut pouvoir
   vérifier que l'optimisation ne casse pas ses capacités phares) :
   - Sous-ensemble figé de **BFCL** (~300 cas, single/multi-turn tool calling,
     scoring déterministe)
   - **Mini bench tool-calling maison** : 30-50 tâches graduées (appel simple,
     choix parmi outils proches, appels chaînés, refus quand aucun outil ne
     convient), outils factices, scoring automatique (JSON valide ? bon outil ?
     bons arguments ?) — reproductible, gratuit et très visuel en vidéo
   - Exclus (trop lourds/bruités pour détecter une dégradation fine) :
     SWE-bench, WebArena, OSWorld… cités comme références à l'écran uniquement
7. **Speculative decoding uniquement** : taux d'acceptation du draft (métriques
   vLLM), **séparément sur les segments de raisonnement et de réponse**, plus
   le test d'identité token-à-token (§ étape C).
8. **Qualitatif** : 30 prompts diversifiés figés, réponses en greedy
   (température 0, comparables entre checkpoints), archivées par étape —
   matériau brut pour la vidéo.

### 6.1 Déterminisme et comparabilité

- Le greedy de vLLM n'est pas bit-à-bit déterministe selon la composition des
  batchs. Fixer la composition, et si nécessaire `--enforce-eager` ou les
  kernels batch-invariants.
- Mesurer et **rapporter la variance** : elle fait partie du résultat.
- Si deux configurations diffèrent de moins de 2 écarts-types, elles sont
  déclarées équivalentes.

### 6.2 Hygiène de mesure Spark

Run de chauffe systématique, mode d'alimentation figé, horodatage et
température ambiante notés, aucun autre processus GPU, un point de mesure
< 30 min.

### 6.3 Le chat template n'est pas universel

Le template du 3.8 est utilisé pour : la génération du dataset, les évaluations
conversationnelles, les prompts qualitatifs, l'inférence finale.
Il n'est **pas** utilisé pour la perplexité (texte brut, §6, point 4).

### 6.4 Manifeste de run (obligatoire)

Chaque run écrit `benchmarks/runs/<run>/manifest.json` : digest de l'image
vLLM, version vLLM, révision HF des modèles, recette de quantization et hash du
set de calibration, `reasoning_effort`, γ, mode MTP, hash du dataset
d'entraînement, commit git, commande exacte. `benchmarks/results.md` agrège
tout le reste et sert de support au tableau final de la vidéo.

---

## 7. Budget GPU H100

| Tâche | Durée estimée |
|---|---|
| Calibration NVFP4 (recette mixte) | ~1-2 h sur 1×H100 |
| Génération dataset par le 27B (B1, tranches S1-S3) | 14-24 h sur 4×H100 |
| Tranche S4 (traces de raisonnement), optionnelle | +4-8 h sur 4×H100 |
| SFT élève 2B (B2) | 6-10 h sur 1×H100 |
| Balayage hors ligne du taux d'acceptation et de γ | ~1-2 h sur 1×H100 |
| Option KD logits | +5 h sur 1×H100 |

**Consommation** : ~75-120 GPU-heures, soit **~20-30 h de mur** en séquentiel
sur le cluster. Sur une fenêtre de 3 jours consécutifs, on se fixe une **règle
de réserve de 50 %** : on ne planifie rien au-delà de ~36 h de mur, le reste
absorbant les itérations, les reprises et les régénérations de dataset.

### Portes go/no-go (pré-enregistrées)

| Porte | Moment | Critère | Si non franchie |
|---|---|---|---|
| **G1** | Fin étape A | Le checkpoint NVFP4 se charge et génère dans vLLM sur GB10, PPL dans la bande attendue | Recette mixte (DeltaNet + ViT en BF16/FP8), puis re-test |
| **G2** | Après B2, sur H100 | Taux d'acceptation ≥ 55-60 % sur le jeu de validation hors ligne | KD sur logits (+5 h) puis plus de données ; si < 40 % après itération → le MTP natif devient le bras final de la vidéo, et l'échec de la distillation devient le chapitre |
| **G3** | Étape D | L'optimisation sort du bruit (> 2 écarts-types) | Rejetée et documentée |

---

## 8. Risques & mitigations

| Risque | Mitigation |
|---|---|
| Tokenizer incompatible entre 3.5 et 3.8 | ✅ Levé : vérifié compatible (§3.4) ; contrat durci et pinné avant l'étape B (§3.5) |
| Le draft distillé ne bat pas le MTP natif | Le MTP est un bras de plein droit (§ étape C) : le résultat est publiable tel quel, et c'est une conclusion honnête et rare |
| vLLM inutilisable sur GB10, ou mode MTP absent de l'image | Étape 0 bloquante : image pinnée par digest, mode MTP identifié avant toute mesure. Le choix « vLLM uniquement » est un objectif d'apprentissage assumé, donc on investit dans l'image, pas dans un second moteur |
| Recette NVFP4 incompatible avec les blocs Gated DeltaNet / le ViT | Recette mixte, couches non standard en BF16/FP8 ; mesurer l'impact réel plutôt que de supposer |
| Le thinking par défaut brouille toutes les mesures et explose le budget dataset | `reasoning_effort` figé et documenté à chaque étape, y compris pour la génération B1 ; nombre de tokens de raisonnement affiché avec le chrono |
| FP8 non supporté sur le GB10 du Spark | Vérifié à l'étape 0 ; sinon on saute au palier NVFP4, aucun impact sur la suite |
| Dégradation peu visible en quantization moderne | La perplexité reste sensible ; sinon c'est un résultat narratif (« robuste jusqu'à 4-bit ») |
| Acceptance rate faible du draft distillé | Mesuré hors ligne sur H100 avant tout run Spark ; porte G2 : KD sur logits, plus de données, ou bascule sur le MTP |
| Items `agentic_coding` non valides (auto-notation) | Réécriture avec tests externes + sandbox **avant** de produire le moindre score (§6, point 5) |
| Blocage bloquant imprévu sur le 3.8 | Fallback global : Qwen3.5-27B (pipeline identique, petits frères natifs disponibles) |
| Calibration contaminée par les prompts d'éval | Vérification automatique d'intersection, hash du set dans le manifeste |
| Dérive de reproductibilité (image, révisions, dataset) | Manifeste de run obligatoire (§6.4) ; repo sous git dès maintenant |

---

## 9. Structure du repo (prévue)

```
fastqwen/
├── .gitignore                   # .env, cache/, checkpoints/, sorties HF
├── .env                         # variables de config (chemins, ports, HF_TOKEN)
├── project-plan.md              # ce fichier
├── check_tokenizer_compat.py    # contrat de compatibilité (exigences dures §3.5)
├── data/
│   ├── prompts/                 # prompts d'évaluation + datasets générés
│   └── calib/                   # set de calibration mixte (FR + code + tools)
├── scripts/
│   ├── generate_dataset.py      # B1 : génération par le prof (vLLM, H100),
│   │                            #      tranches S1-S4 + jeu de validation
│   ├── train_draft.py           # B2 : SFT de l'élève (+ option KD logits)
│   ├── offline_acceptance.py    # C : taux d'acceptation + balayage de γ (H100)
│   ├── quantize_nvfp4.py        # A : FP8 / NVFP4 / palier agressif (ModelOpt,
│   │                            #     llm-compressor, recette mixte de repli)
│   ├── build_calib_set.py       # A : set de calibration mixte + anti-contamination
│   ├── build_eval_datasets.py   # corpus PPL wiki-FR, BFCL subset,
│   │                            # mini tool-calling bench, 30 prompts qualitatifs
│   ├── build_agentic_coding.py  # tests externes pour les 20 items agentic_coding
│   ├── grade_quality200.py      # grader quality-200 (numeric_exact, exec sandboxé,
│   │                            # ifeval_strict — 180 items, manual exclus)
│   ├── verify_identity.py       # C : identité token-à-token greedy vs target
│   └── bench_spark.py           # protocole uniforme (wrapper vllm bench + Prometheus)
└── benchmarks/
    ├── quality/quality-200.jsonl # capacités générales (180 items auto-gradés,
    │                             # 20 manual exclus)
    ├── ppl_corpus/wiki-fr-heldout.txt
    ├── mcq/mmlu_subset.jsonl     # optionnel, référence reconnue à l'écran
    ├── agentique/bfcl_subset.jsonl + toolcalling_mini.jsonl
    ├── prompts/eval_prompts.jsonl
    ├── runs/<run>/manifest.json  # manifeste de chaque run (§6.4)
    ├── samples/<run>/            # sorties greedy par étape
    └── results.md                # tableau cumulé des résultats
```

---

## 10. Références techniques

Points de repère externes pour l'implémentation et pour situer nos chiffres
(ce ne sont pas nos choix de stack : vLLM est le moteur unique du projet).

- Qwen 3.8 — composition de la famille, licences, thinking hybride, contexte,
  architecture dense hybride :
  https://codersera.com/blog/qwen-3-8-model-lineup-2026/
- Qwen3.5-2B (élève) : https://huggingface.co/Qwen/Qwen3.5-2B
- Serving Qwen3.8-27B sur DGX Spark, protocole de benchmarking et A/B documentés :
  https://github.com/amasu/dgx-spark-qwen38
- vLLM + NVFP4 + MTP3 sur DGX Spark :
  https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark
- Checkpoints NVFP4 avec têtes MTP pour sm_121 :
  https://huggingface.co/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121
- Checkpoint AWQ W4A16 (référence de comparaison de format) :
  https://huggingface.co/Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM
- MTP en speculation, mesure de débit :
  https://github.com/julianmb/q38rocm
- Documentation modèle vLLM (Qwen3.8-27B) :
  https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html
