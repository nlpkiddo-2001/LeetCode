"""
================================================================================
GOOGLE L4 — AI/ML ENGINEER: PRACTICAL / "STORY-WRAPPED" DSA QUESTIONS (2026)
================================================================================

The style you're after: real-world-wrapped problems (rooms, loggers, storage,
teleporters, CPUs) where you have to SPOT the underlying pattern (greedy,
topo-sort, BFS/Dijkstra, line sweep, design) instead of it being handed to you.

All of these are from real 2026 Google L4 / DS-MLE interview experiences
(LeetCode Discuss, Blind, CodeZym, InterviewTruth). Your two questions are here:
  - "Room occupancy, 1/2/3 BHK, place people" -> G1 (apartment assignment, greedy)
  - "Dedupe messages within 10s, else send" -> D1 (Logger Rate Limiter + twists)

--------------------------------------------------------------------------------
WHAT THE L4 MLE LOOP ACTUALLY LOOKS LIKE (read this first)
--------------------------------------------------------------------------------
  - The L4 panel is CODING-HEAVY. Multiple reports: the L4 panel is essentially
    DSA. Dedicated ML System Design / ML Theory rounds mostly kick in at L5+.
    BUT some L4 MLE loops still include one "ML domain" round improvised from
    YOUR resume (see Part M) — be ready to actually CODE model bits, not just
    talk design.
  - Typical loop: 1 phone screen (DSA) + 3 onsite DSA + 1 Googleyness & Leadership.
  - No warm-up small talk in coding rounds — they jump straight to the problem.
  - Solve the base problem in ~15-20 min, then spend the rest on FOLLOW-UPS and
    edge cases. That's where Hire -> Strong Hire happens.
  - REAL post-interview feedback that sank a candidate who solved all 3 problems:
    "code lacked readability / not modular — everything in one function."
    => Use helper functions. Name things well. Narrate trade-offs. Handle edges.
  - To advance you generally need to EXCEED expectations in 2 of 3 technical
    rounds, even with a positive Googleyness.
================================================================================
"""


# =============================================================================
# PART G — GREEDY / ASSIGNMENT / SCHEDULING  (your #1 lives here)
# =============================================================================

def g1_assign_students_to_apartments(students, apartments):
    """
    G1. Assign Students to Apartments (share vs. private)   <-- YOUR ROOM QUESTION
    Difficulty: Medium
    Pattern: Greedy assignment (match demand to right unit type)
    Problem: You have apartments of different capacities (e.g. 1BHK / 2BHK / 3BHK,
      i.e. single-room vs multi-room). Students each prefer PRIVACY or are OK to
      SHARE. Students who want privacy should get a single-room unit only if
      necessary; shareable students can fill multi-room units. Assign everyone
      while minimizing wasted single rooms / maximizing occupancy.
    Approach: Sort/bucket by unit size and by preference. Greedily place privacy-
      seekers into the smallest units that satisfy them; pack shareable students
      into multi-room units to capacity. Watch: not enough units, leftover
      capacity, a privacy student forced into sharing (allowed only "if necessary").
    Follow-ups: Minimize number of buildings used; add per-student cost; make it
      an assignment/flow problem if preferences get weighted.
    """

    apartments.sort()

    privacy = []
    share = []

    for student in students:
        if student == "privacy":
            privacy.append(student)
        else:
            share.append(student)

    assign = []

    index = 0

    for student in privacy:
        if index < len(apartments):
            assign.append((student, apartments[index]))
        index += 1

    for student in share:
        if index < len(apartments):
            assign.append((student, apartments[index]))

        index += 1

    return assign



def g2_meeting_scheduling():
    """
    G2. Maximum Non-Overlapping Meetings
    Difficulty: Medium
    Pattern: Greedy interval scheduling
    Problem: Given meeting [start, end) intervals, select the max number that
      don't overlap.
    Approach: Sort by END time (NOT start — classic trap), greedily take each
      meeting starting at/after the last selected end.
    Follow-ups: Minimum rooms to hold all meetings (Meeting Rooms II, heap/sweep);
      weighted intervals -> DP.
    """


def g3_assign_questions_to_volunteers():
    """
    G3. Assign Questions to Volunteers by Skill Tags
    Difficulty: Medium/Hard
    Pattern: Bipartite matching / greedy on tag overlap
    Problem: Each question has tags; each volunteer has skill tags. Optimally
      assign questions to volunteers based on skill-tag vs question-tag match.
    Approach: Model as bipartite graph (volunteer <-> question if tags intersect),
      maximize matches (Hungarian / Hopcroft-Karp, or greedy if constraints allow).
    Follow-ups: Each volunteer max k questions; weighted by overlap size.
    """


def g4_min_cpus_earliest_completion():
    """
    G4. Minimum CPUs for Earliest Task Completion
    Difficulty: Medium
    Pattern: Greedy + heap / sweep over start times
    Problem: Tasks share identical length L; each may start at or after its given
      start time; a CPU runs one task at a time for exactly L. Find the MINIMUM
      number of CPUs so all tasks finish as early as possible.
    Approach: Sweep start times; a min-heap of CPU-free-times; reuse a CPU whose
      task finished, else allocate a new one; track peak concurrency.
    """


def g5_n_cpus_m_tasks_min_time():
    """
    G5. N CPUs, M Tasks — Minimum Completion Time (load balancing)
    Difficulty: Medium/Hard   (Source: Google L4 AI/ML interview experience)
    Pattern: Binary search on the answer + greedy LPT assignment
    Problem: N CPUs, task durations tasks[]. Each CPU runs one task at a time, no
      cooldown. Minimum makespan to finish all tasks?
      e.g. tasks=[5,4,3,2,1], N=2 -> 8  ([5,2]=7 and [4,3,1]=8).
    Approach: Lower bound = max(max(tasks), ceil(sum(tasks)/N)). Binary search T;
      feasibility check: greedily pack longest-first, count CPUs needed <= N.
    Follow-ups: Given that minimum time, find the LEAST N achieving it (binary
      search on N instead of T).
    """


# =============================================================================
# PART D — DESIGN / STREAMING / RATE-LIMITING  (your #2 lives here)
# =============================================================================

def d1_logger_rate_limiter():
    """
    D1. Logger Rate Limiter — dedupe within a time window   <-- YOUR #2 QUESTION
    Difficulty: Easy base, scales with twists
    Pattern: Design + hash map of last-seen timestamp
    Problem: Stream of (message, timestamp). Print/allow a message only if the
      SAME message hasn't appeared in the last 10 seconds; otherwise drop it.
      (Your phrasing: "remove duplicate messages if within 10 seconds, else send.")
    Approach: dict message -> next-allowed-time. Allow iff t >= next_allowed;
      then set next_allowed = t + 10.
    Common interviewer TWISTS (they keep piling these on):
      - Different window per message type / per resourceId.
      - Bound memory: evict stale entries (min-heap or ordered dict by expiry).
      - Out-of-order timestamps in the stream.
      - Concurrency / thread-safety.
      - Return count of dropped messages per window.
    Link: https://leetcode.com/problems/logger-rate-limiter/
    """


def d2_rate_limiter_multi_strategy():
    """
    D2. In-Memory Rate Limiter (multi-strategy)
    Difficulty: Medium/Hard
    Pattern: Design; token bucket / sliding-window log / fixed window per key
    Problem: Implement RateLimiter.isAllowed(resourceId). Each resourceId has its
      own strategy (fixed window, sliding window, token bucket). Return allow/deny.
    Approach: Strategy pattern; per-key state. Discuss accuracy vs memory (sliding
      log is exact but heavy; token bucket is O(1) and smooth).
    Follow-ups: Distributed version (Redis, clock skew); burst handling; cleanup.
    """


def d3_logger_message_printer():
    """
    D3. Logger Message Printer (variant of D1)
    Pattern: Design + hash map
    Problem: Each unique message prints at most once per 10s window (printed at t
      -> blocked until t+10). Essentially D1 with explicit "print" semantics.
    """


def d4_rotated_squares_stream():
    """
    D4. Detect Squares on a Point Stream (ROTATED, any angle)
    Difficulty: Hard
    Pattern: Design + hash map of points + geometry
    Problem: add(point) from a stream; count(queryPoint) = squares formable with
      stored points and the query point. Squares may be rotated at ANY angle
      (harder than the axis-aligned LC "Detect Squares").
    Approach: For each stored point as an adjacent corner, derive the other two
      corners via a 90-degree rotation vector; check existence in the point map.
    """


# =============================================================================
# PART T — TOPOLOGICAL SORT / DEPENDENCY GRAPHS  (very common at L4)
# =============================================================================

def t1_storage_deletion_order():
    """
    T1. Best Order to Delete Storage Parts   (Source: Google L4 DS/MLE onsite)
    Difficulty: Medium
    Pattern: Topological sort (a node deletable only when it has no children)
    Problem: Storage with sub-parts; delete a part only if it has no children.
      Return a valid full deletion order.
      e.g. {A:[B,C]}, {B:[C]}, {C}, {D} -> [D,C,B,A] or [C,D,B,A].
    Approach: DFS post-order topo sort, OR Kahn's BFS on 0-indegree (here: delete
      leaves first).
    Follow-up (asked in the real interview): if two parts are deletable at the
      same time, delete them TOGETHER -> return deletions grouped by "round"
      (BFS level-by-level, all 0-indegree nodes per round).
    """


def t2_compile_packages_multithreaded():
    """
    T2. Compile Packages with Dependencies (multi-threaded)
    Difficulty: Medium
    Pattern: Topological sort by rounds (parallel Kahn's)
    Problem: Dependency graph of packages compiled by multiple threads. Return the
      order packages compile across all parallel rounds.
    Approach: Each round = all currently 0-indegree packages compiled in parallel;
      decrement, repeat. Detect cycles (uncompilable set).
    """


def t3_course_schedule():
    """
    T3. Course Schedule I / II (the plain dependency-graph version)
    Difficulty: Medium
    Pattern: Topological sort / cycle detection
    Problem: Can you finish all courses given prereqs (I)? Return an order (II)?
    Follow-ups: Report the cycle; minimum semesters (parallel rounds, like T2).
    Link: https://leetcode.com/problems/course-schedule-ii/
    """


# =============================================================================
# PART R — GRAPHS: BFS / DIJKSTRA / GRID PATHS
# =============================================================================

def r1_teleporters_with_broken_ones():
    """
    R1. Teleporters with Broken Ones — min cost path   (Source: Google L4 DS/MLE)
    Difficulty: Medium/Hard
    Pattern: 0-1 BFS / Dijkstra (edge cost 0 normally, +1 day to fix a broken one)
    Problem: Country teleport graph. Traveling a normal edge costs 0; if you land
      on / use a BROKEN teleporter you pay 1 day to fix it, then teleport. Find
      min cost source -> destination.
      e.g. edges as adjacency, broken={5}, src=1, dst=6 -> 0 via 1->2->3->6.
    Approach: Build weighted adjacency (0 normal, 1 to leave a broken node), run
      0-1 BFS (deque) or Dijkstra; return dist[dst].
    Follow-ups: Return the path; k broken-fixes budget; undirected variant.
    """


def r2_router_broadcast_and_shutdown():
    """
    R2. Router Reachability on Broadcast-and-Shutdown
    Difficulty: Medium/Hard
    Pattern: BFS on a geometric (Euclidean-range) graph with node consumption
    Problem: Routers have id, (x,y), WORKING/DEFECTIVE. A WORKING router that
      first receives the message rebroadcasts to all WORKING routers within
      `range` (Euclidean), then SHUTS DOWN. DEFECTIVE routers do nothing. From
      source id, does destination id ever receive it?
    Approach: BFS from source; neighbors = WORKING routers within range not yet
      messaged; mark shut down once processed. Return whether dst was reached.
    """


def r3_swim_in_rising_water_variant():
    """
    R3. Swim in Rising Water (variant)   (Source: Google L4 onsite)
    Difficulty: Hard
    Pattern: Dijkstra / binary search + BFS / union-find (minimize the max cell)
    Problem: Grid of elevations; at time t water level is t; you can move to a
      4-neighbor if both cells <= current level. Min time to go top-left ->
      bottom-right (minimize the maximum elevation along the path).
    Approach: Min-heap Dijkstra where path cost = max elevation seen; or binary
      search the threshold + connectivity check.
    Link: https://leetcode.com/problems/swim-in-rising-water/
    """


def r4_pacific_atlantic_water_flow_variant():
    """
    R4. Pacific Atlantic Water Flow (variant)   (Source: Google L4 onsite)
    Difficulty: Medium
    Pattern: Multi-source DFS/BFS from borders (+ can be reframed with DP)
    Problem: Grid of heights; water flows to equal/lower neighbors. Find cells
      that can reach BOTH the top/left ocean and the bottom/right ocean.
    Approach: DFS inward from each ocean's border; intersect reachable sets.
      (Real interview follow-up: redo with BFS + a visited set.)
    Link: https://leetcode.com/problems/pacific-atlantic-water-flow/
    """


def r5_grid_path_with_water():
    """
    R5. Grid Path with Impassable Water Cells
    Difficulty: Easy warm-up -> Medium
    Pattern: BFS/DFS shortest path (or DP for counting)
    Problem: N x N grid, source S, target T, some cells are water (impassable).
      Warm-up often starts as Unique Paths, then adds obstacles and asks shortest
      reachable path.
    Follow-ups: Count paths (DP); weighted terrain (Dijkstra); diagonal moves.
    """


def r6_longest_path_in_grid():
    """
    R6. Longest Path in Grid (enter top row, exit bottom row)
    Difficulty: Hard
    Pattern: DFS backtracking (no cell reuse) — longest simple path is NP-hard
    Problem: Grid 0=empty,1=wall. Enter any empty cell in row 0, exit any empty
      cell in last row, move U/D/L/R through empty cells. Longest travelable path?
    Approach: DFS with visited set + backtracking; discuss why it's exponential
      and what prunes help.
    """


# =============================================================================
# PART L — LINE SWEEP / INTERVALS  (Google leans on this more than most FAANG)
# =============================================================================

def l1_merge_working_hours_timeline():
    """
    L1. Merge Working-Hour Intervals into a Timeline
    Difficulty: Medium
    Pattern: Line sweep (sort events by time)
    Problem: (name, start, end) per person (working at both endpoints). Split the
      timeline into the smallest non-overlapping intervals where the SET of
      working people is constant. Return only intervals with >=1 worker.
    Approach: Emit start/end events, sweep, maintain active-person set, cut a new
      interval whenever the set changes.
    """


def l2_days_everyone_is_free():
    """
    L2. Days When Everyone Is Free
    Difficulty: Medium
    Pattern: Difference array over [1..d] / coverage counting
    Problem: Records "id,start,end" = person unavailable on [start,end]. Return all
      days in 1..d on which EVERY person is free.
    Approach: For each day count how many people are blocked (diff array); a day
      is free iff blocked-count == 0 (careful with per-person double counting).
    """


def l3_size_of_unpainted_segments():
    """
    L3. Size of Unpainted Segments
    Difficulty: Medium
    Pattern: Interval merging / sorted set of painted ranges (sweep)
    Problem: Half-open intervals [start,end) painted in order. For each interval
      return the size of the part NOT already painted by earlier ones.
    Approach: Maintain merged painted ranges (TreeMap/sorted list); for each new
      interval, subtract overlap with existing coverage.
    """


def l4_max_sum_subarray_equal_ends():
    """
    L4. Max Sum Subarray with Equal First and Last Elements
    Difficulty: Medium
    Pattern: Prefix sums + hash map (first index per value)
    Problem: Max sum of a contiguous subarray whose first and last elements equal.
    Approach: prefix[i]; for each value store its earliest index; for a later
      equal value, candidate = prefix[j+1] - prefix[first_index_of_value].
    """


# =============================================================================
# PART S — STRINGS / HASHING (the phone-screen warm-ups they actually gave)
# =============================================================================

def s1_group_rotations():
    """
    S1. Group Strings that are Rotations of Each Other   (Source: L4 MLE phone screen)
    Difficulty: Easy/Medium
    Pattern: Hashing by canonical form
    Problem: ['abc','bca','cab','xyz'] -> [['abc','bca','cab'],['xyz']]. Group all
      strings that are rotations of one another.
    Approach: Canonical key = min rotation (or check t in s+s and equal length).
      Map canonical -> list. Explain complexity and EDGE CASES (feedback here was
      "focus more on edge cases": empty string, single char, duplicates, unequal
      lengths).
    """


def s2_faulty_keyboard_words():
    """
    S2. Faulty-Keyboard Repeated Characters
    Difficulty: Medium
    Pattern: Run-length grouping + dictionary match (Expressive Words variant)
    Problem: A stuck keyboard repeats some chars. Given the typed string and a
      dictionary, return every word the user could have intended.
    Approach: Compress both into (char, count) runs; a dict word matches if same
      char sequence and each typed run count >= word run count (with the stretch
      rule). See LC Expressive Words.
    Link: https://leetcode.com/problems/expressive-words/
    """


def s3_decode_string():
    """
    S3. Decode String   ("3[a2[c]]" -> "accaccacc")
    Difficulty: Medium
    Pattern: Stack (counts + partial strings)
    Follow-ups: recursive parser; malformed input.
    Link: https://leetcode.com/problems/decode-string/
    """


# =============================================================================
# PART TR — TREES (a reported L4 weak-spot round)
# =============================================================================

def tr1_find_leaves_of_binary_tree():
    """
    TR1. Find Leaves of Binary Tree   (Source: Google L4 onsite, round #3)
    Difficulty: Medium
    Pattern: DFS by height-from-bottom (collect nodes level by level from leaves)
    Problem: Repeatedly remove all leaves and record them, until the tree is empty;
      return the list of removed-leaf layers.
    Approach: DFS returning node height = 1 + max(child heights); bucket node
      values by height.
    Link: https://leetcode.com/problems/find-leaves-of-binary-tree/
    """


def tr2_lowest_common_ancestor():
    """
    TR2. Lowest Common Ancestor (binary tree, no BST property)
    Difficulty: Medium
    Pattern: Recursive LCA
    Follow-ups: with parent pointers; deepest-leaves LCA.
    Link: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
    """


def tr3_sum_of_distances_in_tree():
    """
    TR3. Sum of Distances in Tree
    Difficulty: Hard
    Pattern: Rerooting DP (two DFS passes)
    Problem: For every node, sum of distances to all other nodes, in O(n).
    Link: https://leetcode.com/problems/sum-of-distances-in-tree/
    """


# =============================================================================
# PART DP — DYNAMIC PROGRAMMING (common at L4+, "hard DP" shows up)
# =============================================================================

def dp1_partition_equal_subset_sum():
    """
    DP1. Partition Equal Subset Sum (and the "with K changes" variant)
    Difficulty: Medium
    Pattern: Subset-sum 0/1 knapsack DP (bitset speedup)
    Follow-ups: minimum subset-sum difference; allow K element modifications.
    Link: https://leetcode.com/problems/partition-equal-subset-sum/
    """


def dp2_decode_ways():
    """
    DP2. Decode Ways
    Difficulty: Medium
    Pattern: 1D DP; edge cases (leading zeros) are the whole game.
    Link: https://leetcode.com/problems/decode-ways/
    """


def dp3_house_robber():
    """
    DP3. House Robber (I / II circular / III tree)
    Difficulty: Medium
    Pattern: 1D DP -> tree DP.
    Link: https://leetcode.com/problems/house-robber/
    """


def dp4_neighborhood_shuffling():
    """
    DP4. Neighborhood Shuffling   (Source: Google L4 onsite, round #3)
    Difficulty: Hard (reported)
    Pattern: Arrangement/optimization — usually greedy or DP once you find the
      invariant. (Candidate found a better approach only at the very end — so
      spend a minute finding the right framing before coding.)
    Note: Exact statement not public; treat as "rearrange items under adjacency
      constraints to optimize some cost" and practice explaining trade-offs.
    """


# =============================================================================
# PART M — ML DOMAIN ROUND (if your L4 loop includes one; standard at L5+)
# =============================================================================
"""
Reported style: the interviewer INVENTS a system-design problem on the spot from
YOUR resume/past work, then asks you to CODE parts of it. Talking design isn't
enough — you must implement (tensors, a training loop, an optimizer step, a
similarity function). Candidates who only "designed" and then blanked on code
got Lean-No-Hire. Practice writing small model code by hand.

Real / representative prompts:
  - Email/document retrieval + ranking: given an email corpus, a query, and a
    user profile, retrieve and RANK relevant emails.
      -> Dual-encoder (two-tower) transformer: embed email and query separately,
         score by cosine similarity, sort. Follow-up: PERSONALIZE using the user
         profile (concat/gate profile embedding, or re-rank with user features).
      -> Be ready to code: embedding lookup, cosine sim, top-k, a tiny training
         step with an optimizer, and to discuss offline vs online eval metrics
         (recall@k, nDCG), negatives sampling, and latency at serving time.
  - Content moderation system (classification pipeline; precision/recall trade-off,
    human-in-the-loop, threshold tuning, drift).   [L6 report, but same flavor]
  - System over streaming traffic data (features, windowing, online inference).

ML THEORY quick-fire you should be fluent in: bias/variance, regularization,
overfitting fixes, train/val/test leakage, class imbalance, evaluation metrics
(PR-AUC vs ROC-AUC when imbalanced), embeddings & similarity, attention basics,
gradient descent variants, and how you'd debug a model that trains well but
serves poorly.
"""


# =============================================================================
# PART P — HOW TO PRACTICE FOR THIS SPECIFIC STYLE
# =============================================================================
"""
1. For each problem: state brute force -> optimal, then WRITE MODULAR code with
   helper functions and clear names (readability is graded and has sunk people).
2. Solve base in 15-20 min; ask the interviewer clarifying + edge-case questions;
   then invite the follow-up yourself ("can I do better on space?").
3. Drill the pattern buckets above — greedy assignment, topo-sort-by-rounds,
   0-1 BFS/Dijkstra, line sweep, stream/design with expiry, grid DFS/DP.
   These recur far more than any single named problem.
4. Do ~250 medium/hard problems is a commonly reported prep volume, but bias
   toward these practical/story-wrapped variants and timed mocks.
5. If your loop has an ML round: rehearse CODING a two-tower retriever end to end
   from a blank editor. Don't rely on autocomplete/ChatGPT muscle memory.
6. Googleyness (STAR): harsh-but-helpful feedback story, went-out-of-your-way
   story, "what will you learn next year." Deliver them cleanly.

Sources (2026): LeetCode Discuss L4 & DS/MLE experiences, Blind L6 MLE, CodeZym
Google DSA 2026, InterviewTruth, IGotAnOffer, LastRoundAI. Patterns > memorizing;
Google writes original/modified problems on purpose.
"""

if __name__ == "__main__":
    print("Google L4 MLE practical-DSA study set. Pick a pattern bucket and drill.")