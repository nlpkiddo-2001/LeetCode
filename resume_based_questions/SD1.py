"""

QUESTION 1:
At Zoho, whenever a business or ops team needed some data — like how many trial users became paying customers last quarter in a certain region — they had to raise a request with the data team and wait for it.
That team became a bottleneck, and a lot of these questions were actually pretty simple. So I built a system where a non-technical person can just ask a question in plain English and get the SQL for it, without waiting on anyone.

The tricky part isn't writing the SQL — the model can do that easily. The real problem is trust.
If the system gives back a query that looks correct but quietly pulls the wrong numbers, that's worse than giving no answer at all, because the user has no way to know it's wrong.
So the whole design is built around one idea: don't blindly trust the model, and put checks around it.

Here's how a question flows through it. The request first comes into our main service, which runs on CherryPy.
Before doing any real work, it checks a cache with two layers. First is an exact-match cache — if the exact same question was asked before, we just return the stored SQL as-is.
If that misses, there's a similarity cache that finds past questions that mean the same thing but are worded differently.
But here I don't just return the old SQL blindly — I take that stored SQL and make a single LLM call to adjust it to fit the current question.
That's the key part: instead of running the full pipeline, which needs three LLM calls, a similarity hit only needs one. That's a big cut in response time.

If it's a genuinely new question, the next step is figuring out which tables are actually needed.
We don't dump the whole database schema into the prompt — that wastes space and actually makes the model less accurate.
Instead we pick only the relevant tables using three signals together: a small classifier model that scores each table against the question, simple fuzzy text matching on table names, and then an LLM that re-ranks the shortlist using table descriptions and synonyms.
Only the columns and keys from those chosen tables go into the prompt.

That prompt then goes to the actual model — Qwen Coder 7B — which we serve using vLLM.
 We run it at temperature zero so the output is stable and repeatable. The model gives us back a SQL query.

Now comes the part I care about most, which runs before the query touches any database.
I parse the generated SQL and pull out every table, column, and value it's referencing, and I check all of them against the real schema.
If the model made up a column or an enum value that doesn't exist, I catch it right there — every time, because it's a plain lookup, not the model checking itself.
Then another check makes sure the conditions use the right data types.
If there's a small mechanical mistake — like a wrong alias, a slightly-off column name, or a human name that needs to become an internal ID — I fix that automatically instead of asking the model again, which keeps it fast and cheap.
Once the SQL passes all this, we hand it back. The actual running of the query and the access permissions happen in a separate layer that I didn't own.

A few of my choices were deliberate.
I used an off-the-shelf model with good prompting instead of fine-tuning one, because fine-tuning needs a big labeled dataset and separate model versions per schema, and enterprise schemas keep changing.
The tradeoff is the base model is weaker, so I make up for it with strong grounding and heavy validation.
I used three signals for table selection instead of just embeddings, because if we pick the wrong tables the query can never be right, so I wanted more than one safety net.
And I fixed errors with plain deterministic checks instead of looping back to the model, because it's faster and 100% reliable — the downside is it only fixes mechanical mistakes, not logical ones like a wrong join.
One optimization I'm really happy with — for near-duplicate questions, instead of regenerating from scratch with three LLM calls, I reuse the cached SQL and spend just one call adapting it to the new question.
Same correctness, roughly a third of the work.

For results, I tested it on 1,500 real user questions from six months of actual usage, after removing duplicates, and split them into easy, medium, and hard.
I also tracked in production whether the query actually ran successfully and what feedback users gave.
It reached around 93% accuracy and drove roughly a 35% jump in internal adoption once people started trusting the answers.

If I were to take it further, the main thing I'd add is a self-correction loop — feeding validation errors back to the model to fix the logical mistakes my deterministic checks can't catch.
I've actually already built and tested that part; I just haven't switched it on in production yet.
"""



"""
QUESTION 2:
The problem was cost and capacity. We're serving large MoE models to a lot of users, GPUs are expensive and limited, and I needed to push more traffic through the same hardware without latency falling apart.
The key thing I understood early is that LLM inference is limited by memory bandwidth, not raw compute — and my workload was unusual. 
It was very prefill-heavy: think 100,000 tokens going in and only about 2,000 coming out. 
So almost all the work is processing the input, not generating the output. 
That completely changes what you optimize for. 
And the metric that actually matters for the user here is time-to-first-token — how long before they see anything — so my real target was keeping p95 TTFT under 200 milliseconds even when the system is busy.

For the setup: I served MoE models like GLM-4.7 Flash, a 30B Mixture-of-Experts model, and GLM-5.2 in FP8, on H200 and B300 GPUs — the H200 has 141 gigs of memory, the B300 around 283. 
I served them with vLLM. 
On each 8-GPU machine I ran tensor parallel of 8, which splits every layer's math across all eight GPUs, and expert parallel of 8, which spreads the MoE experts across the GPUs so each one only holds a subset. 
No pipeline parallelism, one replica per machine. 
I used FP8 quantized checkpoints — I pulled ready-made ones, I didn't quantize them myself — and FP8 helps two ways: the weights are smaller so there's less memory to move, and that frees up room for a bigger KV cache, which means bigger batches.

Now the actual optimization, which is the interesting part. 
My baseline was basically stock vLLM with default round-robin routing. 
Because I knew the workload was prefill-heavy and the big inputs often shared a lot of common context, the single biggest lever was prefix caching — vLLM reuses the KV cache for a shared prefix instead of recomputing it every time. 
On 100k-token inputs, that's a huge saving. But prefix caching only helps if the request actually lands on the GPU that already has that prefix cached. 
With plain round-robin, it doesn't. 
So I replaced round-robin with AIBrix, which does prefix-aware routing — it looks at the prefix, plus each GPU's cache state, how many requests are running, and how many are waiting, and routes accordingly. 
That's what made the cache actually hit.

Then chunked prefill. 
Normally a 100k-token prefill hogs the whole GPU and blocks everyone else's decode, which spikes time-to-first-token for other users. 
Chunking splits that long prefill into pieces so it interleaves with other requests, which protects TTFT under load. 
On top of that I tuned max_num_batched_tokens and max_num_seqs to keep the GPU fully busy without running out of memory, with GPU memory utilization pinned at 0.92. 
All of this together got me roughly 2.3× the throughput and about double the concurrent users, while still holding p95 TTFT under 200 milliseconds.

For handling overload — I didn't have autoscaling, because we had a limited number of machines, so instead I added rate limiting with two thresholds: requests per second, and tokens per minute. 
I didn't pick those numbers blindly. 
I benchmarked in ramp mode, going from 1 concurrent user up to 32, varying the input and output token sizes to mimic real production traffic, folded that into a one-hour window, and derived the safe thresholds from the actual numbers.

For measurement, I load-tested with vLLM's bench-serve framework, and I built a Grafana layer on top of vLLM's own metrics — throughput, TTFT, latency. 
That's also how I measured volume: every vLLM response reports prompt tokens, generated tokens, and total tokens, so I aggregated that and it came out to around 3.5 billion tokens a week.

A couple of tradeoffs I'd call out. 
FP8 over full precision gave me a big memory and bandwidth win for a small accuracy risk, which I checked held up on our outputs. 
Prefix-aware routing over round-robin massively improved cache hits, but it's more complex and can create hotspots if one prefix gets extremely popular. 
And the lack of autoscaling was a hardware constraint, not a design choice — rate limiting was how I protected the system instead. 
If I had more machines, adding autoscaling and smarter handling of hot prefixes would be the next step.
"""


"""
QUESTION 3:
I built this as a from-scratch training framework to really understand distributed training internals and to have a clean, 
reproducible harness for the whole model lifecycle — pretraining, mid-training or continual pretraining, and fine-tuning — all driven by config instead of code changes. 
A big goal was benchmarking: measuring throughput and how it scales as I change the model size.

The mental model I worked from is that training throughput comes down to two things — 
can the model and its optimizer states fit in GPU memory, and can I keep the GPUs fed so they're never sitting idle. 
Almost every design decision traces back to one of those.

For the setup: I trained GPT-style models — the main one was around 1.2 billion parameters, dim 2048, 20 layers — on [N] H100 GPUs. 
Because a 1.2B model fits comfortably on a single H100, 
I used plain PyTorch DDP. DDP puts a full copy of the model on each GPU, each GPU trains on a different slice of the batch, and then it averages the gradients across GPUs before the optimizer step. 
Since the model fits, DDP is the right choice — simpler and less communication overhead than sharding. 
If the model didn't fit on one GPU, that's when I'd move to FSDP.


Here's how a run works. 
Everything is defined in a YAML file — model size, batch size, learning rate, precision, optimizer, schedule. 
I launch with torchrun, which spins up one process per GPU. 
Each process loads its own shard of the data through a distributed data loader, so no two GPUs see the same batch. 
Training runs in BF16 mixed precision — I use BF16 specifically over FP16 because BF16 has the same exponent range as full precision, 
so I don't need loss scaling and I don't hit the overflow problems FP16 has. 
I also turn on torch.compile and set the matmul precision to high, which lets the H100 use its faster tensor-core math — that's basically free throughput. 
To simulate a large batch without needing more memory, I use gradient accumulation: run several forward-backward passes, accumulate the gradients, then do one optimizer step. 
For the optimizer I support both AdamW and Muon, and for the learning rate I support cosine and Warmup-Stable-Decay — 
WSD is nice for continual training because the long stable phase means you can branch or resume without the learning rate having already decayed away.

The part I'm happiest about engineering-wise is checkpointing and auto-resume. Long training runs crash — a node dies, something OOMs — and you can't afford to restart from zero. 
So I save [weights, optimizer state, scheduler state, and the step count] and the pipeline auto-detects the latest checkpoint and resumes exactly where it left off. 
I also rotate checkpoints, keeping only the last three, because these files are huge and disk fills up fast — that's a straight tradeoff between crash-safety and storage. 
And the multi-stage pipeline chains automatically: the pretrain stage writes a checkpoint, mid-training auto-detects it and continues, then fine-tuning picks up from there.

For measuring, I tracked tokens per second and computed Model FLOPs Utilization — how much of the GPU's theoretical peak I was actually using — and got it to around 40%, which is a solid number for a setup like this. 
I benchmarked how throughput scaled as I changed model size, and I tracked downstream eval accuracy against reference baselines so I knew the framework was training correctly, not just fast.

A few tradeoffs I'd call out. DDP over FSDP: correct here because the model fits, and it's simpler — but it doesn't scale to models bigger than one GPU, which is the first thing I'd change to go larger. 
BF16 over FP16: same range as FP32 so no loss-scaling headache, for a tiny precision cost. 
Gradient accumulation over a genuinely bigger batch: lets me get large-batch stability on limited memory, at the cost of more forward-backward passes per step. 
And keeping only three checkpoints: saves disk, but if I needed to analyze training history further back, I'd lose it.

If I took it further: FSDP so I can train models that don't fit on a single GPU, overlapping the gradient communication with computation to push MFU higher, and adding more thorough eval hooks during training instead of after."
"""



"""
=============================================================================
SYSTEM DESIGN — Real-Time Voice AI Agent for a Bank
Millions of calls/day · <500ms turn latency · barge-in · Indian languages · 99.99%
=============================================================================

00. HOW TO FRAME IT (say this first)
------------------------------------
This LOOKS like a chatbot but it isn't. It's a REAL-TIME STREAMING MEDIA
system with an ML backend. The hard part isn't the AI being smart — it's
turning a spoken turn around in <500ms, while people talk over it, in a dozen
languages, without ever going down.

Three constraints drive every decision:
    (1) sub-500ms latency   (2) barge-in   (3) banking-grade reliability + compliance

Clarify scope up front: inbound only, task-bounded self-service (balance,
statement, card block, transfers) with human fallback, and we ASSEMBLE
best-in-class speech/LLM services and own the orchestration.


01. REQUIREMENTS
----------------
Functional : answer call, transcribe live, understand intent, reply in natural
             voice, allow interruption, multilingual, authenticate, execute
             banking actions, escalate to human, record + audit.

Non-functional (where the real design lives):
    Latency      <500ms P50, <800ms P95 (end-of-speech -> agent speaks)
    Availability 99.99% (~52 min/yr) — a live call can't be retried later
    Scale        ~5M calls/day -> ~35-40k concurrent at peak
    Accuracy     low WER across accents/languages; near-ZERO wrong transactions
    Compliance   RBI data localization + PCI — a DESIGN DRIVER, not a footnote

Note: phone audio is narrowband 8kHz — harder for ASR than clean mic audio.
Models must be tuned for telephony, accents, and code-mixing.


02. CAPACITY MATH (concurrency is the number everything hangs off)
------------------------------------------------------------------
    avg concurrent = (5,000,000 calls x 180s) / 86,400s ~= 10,400
    peak factor ~3.5x (mornings, salary day, month-end, festivals)
    => design hot path for ~35-40k CONCURRENT sessions

Key insight: not every call uses every resource every second.
    ASR  -> runs continuously (whole call)
    TTS  -> only while agent speaks (~40% of call)
    LLM  -> only during a turn
Size each GPU pool to its DUTY CYCLE, not to raw call volume. Keeps cost sane.


03. THE 500ms CRUX (this beat wins the interview)
-------------------------------------------------
Naive design = sequential pipeline: transcribe fully -> LLM fully -> synth fully
-> play. That's 1.5-2s. Budget blown 3x over.

FIX: every stage STREAMS and stages OVERLAP in time.
    - ASR emits partial transcripts continuously
    - LLM starts composing on the PARTIAL, before caller even finishes
    - TTS voices the first words before LLM finishes generating the rest
    - You optimize TIME-TO-FIRST-AUDIO, not total processing time

Rough budget (end-of-speech -> first agent audio):
    Endpointing (VAD) .. ~150ms  <- biggest + most controllable; make it adaptive
    ASR finalize ....... ~45ms
    LLM time-to-1st-tok  ~180ms  <- warm models, KV-cache, small fast model
    TTS 1st audio chunk  ~90ms   <- streaming vocoder, phrase by phrase
    Network + jitter ... ~35ms
                         ------
                         ~500ms  (tight — which is WHY overlap matters)

Nuance to impress: part of that 500ms is silence the CALLER controls. And if a
bank API is slow, play a tiny filler ("let me pull that up") — buys time AND
feels more human.


04. ARCHITECTURE — split into TWO PLANES (the core structural decision)
----------------------------------------------------------------------
MEDIA PLANE (stateful, sticky, real-time audio — "the plumbing"):
    SBC + SIP trunks ...... switchboard to phone network; multi-carrier for HA
    Media gateway ......... RTP audio, jitter buffer, echo cancel (AEC), VAD
    Session orchestrator .. the "brain" of ONE call: fans audio to ASR, feeds
                            LLM, streams TTS back, handles barge-in
    Redis ................. fast + replicated session state
    

INFERENCE PLANE (GPU-heavy, load-balanced — "the thinking"):
    Streaming ASR + language ID
    LLM + RAG + tool-calling
    Streaming TTS
    Model serving (batching, autoscale)

ENTERPRISE (behind everything):
    Core banking (system of record) · Auth/biometrics · Fraud engine ·
    Human agents · Logging/audit

WHY split the planes: media is stateful + sticky to a call (scales on concurrent
sessions); inference is stateless per request (load-balance + batch it). They
scale and FAIL differently, so keep them separate.


05. ONE TURN, END TO END
------------------------
Call lands -> orchestrator spins up session actor + loads state -> audio streams
in (RTP) -> gateway does AEC + runs VAD continuously -> ASR emits partials +
language ID tags language -> endpointing fires -> transcript finalizes -> LLM
resolves intent, pulls grounded facts (RAG), decides answer-vs-tool-call ->
tokens stream to TTS -> audio chunks stream back to caller (agent speaks inside
budget) -> VAD keeps listening for barge-in -> recordings + audit written AFTER,
never on the hot path.


06. STREAMING PIPELINE (deep dive)
----------------------------------
Everything hot is a STREAM, not a request: audio in, growing partial transcripts,
LLM token stream, TTS audio-chunk stream. Streaming is what unlocks overlap AND
interruptibility.
Watch-out: ASR partials are REVISABLE — don't fire a banking tool on unstable
partials; commit tools only after finalization.


07. BARGE-IN (deep dive #1 — trickiest, most impressive)
--------------------------------------------------------
Problem: while agent speaks, its OWN voice echoes back down the line; naive VAD
thinks the caller is talking.

    1. Acoustic Echo Cancellation (AEC) subtracts the known outgoing audio
       -> what's left is basically just the caller
    2. Full-duplex VAD runs on cleaned audio EVEN WHILE TTS plays
    3. Caller speech sustained ~200ms (ignore "mm-hmm" backchannels)
       -> fire barge-in event
    4. Orchestrator INSTANTLY kills TTS, flushes outbound buffer, cancels
       in-flight generation -> switch to listening -> re-plan

Product judgment: not every interruption stops the agent. Short "yeah ok" is a
backchannel. Tuning that threshold = natural vs twitchy.


08. INDIAN LANGUAGES (deep dive #2 — region-specific)
-----------------------------------------------------
Not just 12 languages — constant CODE-MIXING ("mera balance kitna hai, transfer
5000 to savings").

    - Prefer ONE multilingual ASR model (telephony-tuned, code-mix trained) over
      one-model-per-language — handles mid-sentence switching + simpler ops
    - Detect language in first few hundred ms, per-utterance (people switch)
    - Use a natively multilingual LLM -> AVOID translate-to-English pivot
      (extra latency hop + lost nuance)
    - TTS: per-language voices but ONE consistent brand persona

Failure mode to volunteer: numbers/amounts/account-IDs break most dangerously.
Always READ BACK + CONFIRM before executing. Accuracy on entities > fluency.


09. DIALOG SAFETY — money can't be a free LLM
---------------------------------------------
HYBRID design:
    Informational (FAQ, "explain this charge") -> LLM-driven, grounded via RAG
    Sensitive (auth, transfers, card block)    -> LLM only ROUTES into a fixed,
                                                  deterministic, auditable flow

Banking actions = typed tools (getBalance, blockCard, transfer). LLM proposes;
a POLICY LAYER validates BEFORE execution (auth level, limits, fraud score).
Money movement always needs explicit spoken confirmation. Treat caller speech as
UNTRUSTED input -> can't override policy (prompt-injection defense).


10. SCALING INFERENCE
---------------------
    - Plane separation lets media + inference scale independently (foundation)
    - Continuous batching for LLM -> pack many short generations per GPU pass
    - Right-size the model: small fine-tuned model for the fast path, big model
      reserved for hard cases
    - Quantization (INT8/FP8) + optimized serving runtime
    - Pre-warm capacity for known peaks (cold GPU spin-up is slow)
    - Autoscale on active sessions + queue depth (not just CPU)
    - Backpressure: if saturated, queue/hold new calls rather than accept ones
      you can't serve inside the latency budget


11. HIGH AVAILABILITY
---------------------
Harder than a web service — a live call is stateful, can't be retried.
    - Redundant carriers + SIP trunks
    - Multi-AZ, multi-region WITHIN India (localization)
    - Stateless services + replicated Redis -> any orchestrator replaceable
    - In-flight calls can't seamlessly live-migrate -> drain gracefully on deploy
    - TIERED graceful degradation:
        LLM degraded  -> constrained menu flow
        ASR struggles -> keypad (DTMF) entry
        worst case    -> route to human
      Degrade in STEPS, don't fail hard.
Honesty scores: don't claim "zero dropped calls ever" — say "a failure drops at
most a small set of calls + offers instant human fallback."


12. SECURITY & COMPLIANCE (design driver)
-----------------------------------------
    - RBI data localization -> customer/payment data stays in India (constrains
      regions + vendors up front)
    - PCI-DSS for any card-data path -> minimize, tokenize, isolate
    - Encryption everywhere: SRTP (media), TLS (services), at-rest (recordings)
    - Layered auth: voice biometrics (+ liveness/anti-spoof) + OTP + KBA,
      escalating with transaction risk
    - Redact PII from logs/transcripts; tamper-evident audit trails
    - Fraud + anomaly detection (phone channel = favorite for social engineering)


13. STATE & STORAGE
-------------------
    Live session context ... in-memory + Redis (sub-ms, replicated)
    Customer/account data .. core banking = SYSTEM OF RECORD (never duplicated)
    RAG knowledge base ..... vector DB + doc store
    Call recordings ........ object storage, tiered (~7TB/day raw, cold after N days)
    Transcripts + audit .... searchable store/warehouse (QA, disputes, analytics)

Golden rule: system of record STAYS the core banking platform. Voice system
holds only ephemeral session state + derived artifacts.


14. OBSERVABILITY & METRICS
---------------------------
    Engineering : turn latency P50/P95/P99, time-to-first-token,
                  time-to-first-audio, barge-in reaction time, dropped-call rate,
                  GPU utilization + queue depth
    Product     : ASR WER (per language/accent), CONTAINMENT RATE (resolved w/o
                  human), escalation rate, task success, false barge-ins, CSAT,
                  and critically TRANSACTION ERROR RATE
    Plus distributed tracing across the audio pipeline + sampled recordings for QA.


15. TL;DR (say this even if out of time)
----------------------------------------
Real-time streaming media system with an ML backend, split into a stateful media
plane and a GPU inference plane. Hit 500ms by STREAMING + OVERLAPPING every stage.
Barge-in = echo cancellation + always-on VAD that instantly stops the agent.
Indian languages = one unified telephony-tuned multilingual model with entity
read-back. Money moves only through deterministic, confirmed, auth-gated flows.
Scale = plane separation + continuous batching + duty-cycle-aware GPU pools.
Availability = redundant carriers + tiered graceful degradation + human fallback.
RBI + PCI compliant with data localization from day one.
=============================================================================
"""