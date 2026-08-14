"""
================================================================================
 eBay - AI PLATFORM ENGINEER  |  COMPLETE QUESTION BANK  (Bengaluru, India)
================================================================================

Role ref  : R0074429 (listing closed)  |  Compiled: Aug 2026
Sources   : eBay candidate reports (LeetCode/Glassdoor/Blind/Naukri/CodingKaro),
            eBay MLE/AI-Engineer interview guides, and role-typical MLOps/
            platform question sets.

LEGEND
    [REPEATED] = reported by multiple eBay candidates (highest priority)
    [EXPECTED] = strongly implied by the role + eBay's loop structure
    [STRETCH]  = deeper / senior / performance-eng flavour
    (LC ...)   = maps to a known LeetCode-style pattern

WHY THIS SHAPE
    "AI Platform Engineer" = infra/MLOps role. eBay's loop for AI/ML roles mixes
    FOUR traditions: SWE coding (DSA), ML theory/DL, ML+platform system design,
    and MLOps/production. This file covers all four + SQL + behavioral.
================================================================================
"""


def dsa_repeated_questions():
    """
    ---------------------------------------------------------------------------
    SECTION 1 : DSA / CODING  (repeated + expected)
    ---------------------------------------------------------------------------
    Format : CodeSignal / HackerRank OA + live rounds. Often 2-3 problems in one
             hour; some rounds report up to 3 coding problems back-to-back.
    Mix    : ~26% Easy, ~68% Medium, ~5% Hard. Emphasis on ARRAY + HASH TABLE.
    Style  : narrate approach + complexity BEFORE coding; brute-force -> optimal.

    [REPEATED] ACTUAL QUESTIONS FROM eBay CANDIDATES
        1.  Number of Islands - grid/graph traversal.              (LC 200)
        2.  Reverse Pairs: count pairs (i<j) where nums[i] > 2*nums[j].
            (merge-sort / BIT based - LC 493, Hard)
        3.  Priority Queue + HashMap combined problem.             (LC Hard)
        4.  Move all ODD numbers to front, EVEN to back, PRESERVE order,
            IN-PLACE (no extra mem), O(n) time.  <-- classic eBay twist
        5.  Merge K sorted arrays / lists.                         (LC 23)
        6.  Maximum Value at a Given Index in a Bounded Array.     (LC 1802)
        7.  Invalid Transactions.                                  (LC 1169)
        8.  Zigzag Conversion.                                     (LC 6)
        9.  Generics / type-safe code (Java) - write a generic container.
        10. Data-structure design + DB schema design (manager round).

    [EXPECTED] PATTERNS TO DRILL (eBay leans on these)
        - Hash map: frequency, grouping, two-sum family, dedup.
        - Heap: top-K, merge-K, streaming median, K closest.
        - Graph: BFS/DFS, number of islands, topo sort, cycle detect,
          shortest path.
        - Trees: traversals, BST ops, LCA.
        - Sliding window / two pointers: longest substring, max window.
        - Sorting + custom comparators; in-place array manipulation.
        - DP: classic 1D/2D, edit distance, coin change.
        - Concurrency / thread-safe design primitives (occasionally).
        - LRU / LFU cache (ties to serving + caching).            (LC 146/460)
        - Rate limiter (token bucket / sliding window).

    [EXPECTED] eBay-SPECIFIC FORMATS (prep these, not just LeetCode)
        - Debug an existing codebase: find bug -> safe fix -> add tests ->
          explain complexity + tradeoffs live.
        - Complete missing functions in a provided repo.
        - Write unit tests for given code.
        - OOP under time pressure: design classes for a parking lot / booking
          system scaled to MILLIONS (candidates were dinged for ignoring OOP).
    ---------------------------------------------------------------------------
    """
    pass


def ml_theory_and_deep_learning():
    """
    ---------------------------------------------------------------------------
    SECTION 2 : ML THEORY + DEEP LEARNING  (concept round)
    ---------------------------------------------------------------------------
    eBay AI/ML rounds test fundamentals even for platform roles - you must speak
    the ML teams' language. Expect a technical director or MLE probing depth.

    [EXPECTED] CORE ML THEORY
        - Explain gradient descent (batch vs SGD vs mini-batch).
        - Bias-variance tradeoff; how it shows up in practice.
        - Overfitting: causes + regularization (L1/L2, dropout, early stop).
        - Precision/recall, F1, ROC-AUC - when each matters (fraud = imbalanced!)
        - Handling class imbalance (resampling, class weights, thresholding).
        - Cross-validation; data leakage and how to avoid it.
        - Feature scaling / encoding; why it matters for which models.
        - Evaluation metric choice tied to business goal (NDCG for ranking,
          catch-rate/false-positive for fraud).

    [EXPECTED] DEEP LEARNING
        - Explain backpropagation to a junior engineer.
        - ReLU vs sigmoid in hidden layers - why ReLU.
        - Vanishing gradient problem + two fixes.
        - Self-attention and its O(n^2) cost.
        - "Validation loss drops but validation accuracy is flat" - hypotheses.
        - Batch norm / layer norm - what + why.

    [REPEATED] eBay HANDS-ON ML CODING (this actually happened)
        - Colab notebook: COMPLETE a feed-forward network in PyTorch - build
          layers, implement BACKPROP, and INFERENCE. (First "algo" round turned
          out to be pure ML - be ready even if recruiter said CodeSignal.)
        - Python: preprocess data -> train a basic model -> evaluate accuracy
          (implement the functions live, ~1 hr).
        - In-depth resume/project walkthrough FROM AN ML PERSPECTIVE - take them
          through the FULL dev process (data -> features -> model -> eval ->
          deploy), with SPECIFIC metrics (NDCG lift, fraud catch rate, latency).

    [STRETCH] PERFORMANCE / LOW-LEVEL (reported in later rounds)
        - Model QUANTIZATION tradeoffs (accuracy vs latency vs size).
        - Manual CALIBRATION techniques.
        - GPU KERNEL FUSION - have you implemented it?
        - Writing parametric mathematical curves in C++ (abstract coding task).
    ---------------------------------------------------------------------------
    """
    pass


def technical_ml_production():
    """
    ---------------------------------------------------------------------------
    SECTION 3 : TECHNICAL ML - PRODUCTION / MLOps  (the platform core)
    ---------------------------------------------------------------------------
    This is where an AI PLATFORM hire is really judged. Concept + practical.

    [REPEATED] PRODUCTION IMPLEMENTATION QUESTIONS
        - How would you BUILD, DEPLOY, and OPTIMIZE an ML workflow in a real
          engineering environment? (most of the round is spent here)
        - AI application ARCHITECTURE - how ML systems are designed + integrated.
        - Model training + inference: production implementation challenges.

    [EXPECTED] MLOps FUNDAMENTALS (know the "why," not just tool names)
        - Training-serving skew: what it is, prevent + detect.
        - Data drift vs concept drift: monitor each; alerting.
        - Detect model DEGRADATION WITHOUT real-time ground-truth labels.
          (proxy metrics, prediction-distribution monitoring, delayed labels)
        - CI/CD for ML vs normal software: data + model versioning, eval gates,
          retraining triggers, rollback.
        - "Offline metrics good, production underperforms" - debug end to end.
        - Model versioning + rollback strategy in prod.
        - What belongs in a model registry; why you need one.
        - Batch vs real-time (online) inference tradeoffs.
        - Reproducibility: guarantee a model can be rebuilt bit-for-bit.
        - Autoscaling inference: which metrics you scale on; GPU scheduling.
        - Canary vs blue-green vs shadow deployment - when to use each.
        - Secrets handling, least privilege, data boundaries/governance.
        - Incident handling for a degraded prod model (runbook thinking, SLOs).

    TOOLING (be conversational; go DEEP on 1-2)
        Serving       : KServe, Triton, TorchServe, Seldon
        Orchestration : Airflow, Argo Workflows, Kubeflow, Flyte, Dagster
        Registry/track: MLflow, model registry concepts
        Infra         : Kubernetes, Docker, Terraform/IaC, GCP/AWS
        Observability : Prometheus, Grafana, drift-monitoring stacks
    ---------------------------------------------------------------------------
    """
    pass


def system_design_questions():
    """
    ---------------------------------------------------------------------------
    SECTION 4 : SYSTEM DESIGN  (highest-leverage round for this role)
    ---------------------------------------------------------------------------
    Two flavours show up: (A) ML/infra platform design, (B) classic distributed
    system design. Prep BOTH. Interviewer challenges scalability + reliability,
    especially real-time inference + A/B infra. Go deep, justify every decision.

    [REPEATED] eBay-REPORTED SYSTEM DESIGN PROMPTS
        - Design a Dropbox-like system (file sync/storage). Justify each
          component; expect deep follow-ups on every part.
        - Design an END-TO-END ML-powered service: real-time SEARCH RANKING,
          ADS RELEVANCE, or FRAUD DETECTION.
              -> two-stage arch: candidate generation (retrieval) + ranking
              -> specify what runs ONLINE vs OFFLINE with clear boundaries
              -> data ingestion, feature store, training cadence, online serving
              -> latency budget + QPS, freshness needs
              -> experimentation / rollout (A/B), fallback ranking, circuit
                 breakers as failure modes.

    [EXPECTED] ML-INFRA / PLATFORM DESIGN (map directly to the role)
        1. Design a MODEL SERVING platform: low-latency real-time inference,
           autoscaling, GPU scheduling, canary/blue-green/shadow, rollback.
        2. Design a TRAINING / RETRAINING pipeline: orchestration, artifact +
           model registry, versioning, reproducibility, scheduled retrain.
        3. Design a FEATURE STORE: offline vs online consistency, point-in-time
           correctness, training-serving skew prevention.
        4. Design a MODEL MONITORING / DRIFT system: prediction-distribution
           tracking, data vs concept drift, auto-retrain triggers, alerting.
        5. Design a MULTI-TENANT ML platform: team isolation, RBAC, quotas,
           GPU bin-packing, cost attribution, self-service for DS teams.
        6. Design an A/B TESTING framework to safely evaluate new model versions.
        7. Design a real-time RECOMMENDATION system for the e-commerce homepage.
        8. Design real-time FRAUD DETECTION with a ~100ms latency budget.

    ALWAYS PROBED (bake these into every answer)
        - requirements -> scale/constraints (QPS, latency, freshness)
        - high-level arch -> deep dive 1-2 components
        - failure modes: skew, drift, reproducibility loss, fallback paths
        - latency vs cost vs reliability tradeoffs; SLOs/SLIs + instrumentation
        - rollback + incident handling + runbooks
        - data governance/security around training data
        - "what changes at 10x scale?"

    [STRETCH] SENIOR SIGNALS
        - GPU fleet utilization / bin-packing; multi-region serving + residency;
          per-team/per-model cost attribution; online eval infra.
    ---------------------------------------------------------------------------
    """
    pass


def sql_and_data_modeling():
    """
    ---------------------------------------------------------------------------
    SECTION 5 : SQL + DATA MODELING  (common stumble point)
    ---------------------------------------------------------------------------
    eBay marketplace = massive event tables (impressions, clicks, purchases
    across 2B+ listings). Reported as a round where notebook-heavy candidates
    lose the offer. Budget real prep time here.

    [EXPECTED]
        - Fluent WINDOW FUNCTIONS: ROW_NUMBER, RANK, LAG/LEAD, running totals,
          partitioned aggregates.
        - Complex JOINS across large event tables under time pressure.
        - Sessionization from event streams (gap-based windows).
        - Funnel / conversion queries (impression -> click -> purchase).
        - Top-N per group; dedup; de-duplicating late-arriving events.
        - Data modeling: design schema for events/features; normalization vs
          denormalization tradeoffs for analytics + serving.
    ---------------------------------------------------------------------------
    """
    pass


def behavioral_questions():
    """
    ---------------------------------------------------------------------------
    SECTION 6 : BEHAVIORAL / TEAM FIT
    ---------------------------------------------------------------------------
    eBay values "impact and integrity." STAR, 2-3 min each: 1-line setup ->
    what YOU did ("I", not "we") -> measurable RESULT -> lesson. Note: a manager
    round sometimes replaces coding with 3 SITUATIONAL questions - be ready.

    [EXPECTED]
        - Handle conflicting priorities between teams (you serve DS + product
          eng at once - very likely).
        - A production incident you resolved + what you changed after.
        - A tradeoff under tight constraints (time/cost/quality).
        - A time you disagreed with a decision - how you handled it.
        - A time you improved reliability / cut cost / cut latency (metrics!).
        - A model/system that underperformed in production - what you did.
        - A difficult stakeholder.
        - Why eBay? Why this platform role? How do you make ML engineers faster?

    PREP CHECKLIST
        [ ] cross-team conflict story
        [ ] incident / on-call story with concrete fix
        [ ] measurable-impact story (NUMBERS)
        [ ] failure/lesson story
        [ ] "why eBay + why platform" narrative
    ---------------------------------------------------------------------------
    """
    pass


def questions_to_ask_them():
    """
    ---------------------------------------------------------------------------
    SECTION 7 : SMART QUESTIONS TO ASK  (scored signal for platform roles)
    ---------------------------------------------------------------------------
        - What does the ML platform team own end-to-end today vs on the roadmap?
        - How many models / teams does the platform serve, at what scale/QPS?
        - Biggest current reliability or cost pain point?
        - Build vs buy philosophy for serving / orchestration?
        - How is on-call structured for platform incidents?
        - How do you measure the platform's success (DS productivity? SLOs?)?
        - What does the first 90 days look like?
    ---------------------------------------------------------------------------
    """
    pass


def prep_plan():
    """
    ---------------------------------------------------------------------------
    SECTION 8 : PRIORITIZED 2-WEEK PREP PLAN
    ---------------------------------------------------------------------------
    EFFORT RANK FOR THIS ROLE
        1. System design (ML infra + one distributed classic like Dropbox).
        2. DSA on CodeSignal, TIMED + debugging/writing tests in a codebase.
        3. MLOps/production fluency + 1-2 tools deep (Section 3).
        4. SQL window functions + large-table joins (Section 5).
        5. ML/DL fundamentals refresher (Section 2) + PyTorch FFN from scratch.
        6. Behavioral STAR stories with metrics (Section 6).

    WEEK 1 : Sections 1 + 2 + 5 (coding, ML/DL, SQL) - hands-on reps daily.
    WEEK 2 : Sections 3 + 4 (MLOps + system design) out loud + mock interviews.
    NIGHT BEFORE : re-read Sections 3 & 4; rehearse design method + STAR;
                   prep 3-4 questions from Section 7.

    GROUND YOUR ANSWERS
        Skim eBay's tech blog on search / recommendation / fraud infra at
        130M+ buyer, 2B+ listing scale so system-design answers feel real.
    ---------------------------------------------------------------------------
    """
    pass


if __name__ == "__main__":
    for section in (
        dsa_repeated_questions,
        ml_theory_and_deep_learning,
        technical_ml_production,
        system_design_questions,
        sql_and_data_modeling,
        behavioral_questions,
        questions_to_ask_them,
        prep_plan,
    ):
        print(section.__doc__)