# Master Theorem

The **Master Theorem** helps us find the time complexity of recursive algorithms that have this form:

$$
\boxed{T(n)=aT\left(\frac{n}{b}\right)+f(n)}
$$

Think of it as:

> **How many recursive subproblems?** + **How much work outside recursion?**

---

## 1. What do `a`, `b`, and `f(n)` mean?

Consider:

```text
function example(n):
    example(n / 2)
    example(n / 2)

    for i = 1 to n:
        print(i)
```

There are **2 recursive calls**:

```text
example(n / 2)
example(n / 2)
```

And each recursive call works on `n/2`.

The loop does `n` work.

Therefore:

$$
T(n)=2T(n/2)+n
$$

| Variable | Meaning                       | Here |
| -------- | ----------------------------- | ---: |
| `a`      | Number of recursive calls     |    2 |
| `b`      | Factor by which input shrinks |    2 |
| `f(n)`   | Work outside recursion        |  `n` |

---

## 2. The Most Important Calculation

Once you have:

$$
T(n)=aT(n/b)+f(n)
$$

calculate:

$$
\boxed{n^{\log_b a}}
$$

This is the **benchmark** that you compare `f(n)` against.

Think of it as:

> **Recursive work benchmark = $n^{\log_b a}$**

Then compare:

$$
f(n)
$$

against:

$$
n^{\log_b a}
$$

That's essentially the whole Master Theorem.

---

## 3. The Three Cases

There are three possibilities.

### Case 1 — `f(n)` is Smaller

If:

$$
f(n) < n^{\log_b a}
$$

by a polynomial factor, then:

$$
\boxed{T(n)=\Theta(n^{\log_b a})}
$$

In simple terms:

> **Recursive calls dominate.**

---

### Case 2 — They Are the Same Size

If:

$$
f(n)=\Theta(n^{\log_b a})
$$

then:

$$
\boxed{T(n)=\Theta(n^{\log_b a}\log n)}
$$

In simple terms:

> **Both contribute equally.**

---

### Case 3 — `f(n)` is Larger

If:

$$
f(n) > n^{\log_b a}
$$

by a polynomial factor **and the regularity condition holds**, then:

$$
\boxed{T(n)=\Theta(f(n))}
$$

In simple terms:

> **The work outside recursion dominates.**

---

## 4. An Easy Way to Remember

Imagine a competition:

$$
\boxed{n^{\log_b a}}
\quad\text{vs}\quad
\boxed{f(n)}
$$

### Recursive Side Wins

$$
f(n) < n^{\log_b a}
$$

→ **Case 1**

$$
T(n)=\Theta(n^{\log_b a})
$$

### Tie

$$
f(n)=\Theta(n^{\log_b a})
$$

→ **Case 2**

$$
T(n)=\Theta(n^{\log_b a}\log n)
$$

### `f(n)` Wins

$$
f(n)>n^{\log_b a}
$$

→ **Case 3**

$$
T(n)=\Theta(f(n))
$$

---

## 5. Example: Case 1

Suppose:

$$
T(n)=2T(n/2)+1
$$

### Step 1: Identify Values

$$
a=2,\quad b=2
$$

$$
f(n)=1
$$

### Step 2: Calculate Benchmark

$$
n^{\log_b a}
$$


$$
n^{\log_2 2}
$$


### Step 3: Compare

$$
f(n)=1
$$

versus:

$$
n
$$

Clearly:

$$
1<n
$$

So recursive work dominates.

**Case 1:**

$$
\boxed{T(n)=\Theta(n)}
$$

---

## 6. Example: Case 2

Now:

$$
T(n)=2T(n/2)+n
$$

Again:

$$
a=2,\quad b=2
$$

and:

$$
f(n)=n
$$

Benchmark:

$$
n^{\log_2 2}=n
$$

Compare:

$$
f(n)=n
$$

with:

$$
n^{\log_b a}=n
$$

They are equal.

Therefore **Case 2**:

$$
\boxed{T(n)=\Theta(n\log n)}
$$

---

## 7. Example: Case 3

Consider:

$$
T(n)=2T(n/2)+n^2
$$

We have:

$$
a=2,\quad b=2
$$

$$
f(n)=n^2
$$

Benchmark:

$$
n^{\log_2 2}=n
$$

Compare:

$$
n^2>n
$$

So the non-recursive work is larger.

This is **Case 3**, giving:

$$
\boxed{T(n)=\Theta(n^2)}
$$

---

## 8. The Pattern You Should Memorize

Look at these three:

| Recurrence    | Benchmark | Case | Answer            |
| ------------- | --------- | ---- | ----------------- |
| $2T(n/2)+1$   | $n$       | 1    | $\Theta(n)$       |
| $2T(n/2)+n$   | $n$       | 2    | $\Theta(n\log n)$ |
| $2T(n/2)+n^2$ | $n$       | 3    | $\Theta(n^2)$     |

Notice what's changing:

```text
             Recursive benchmark
                     n
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
       1             n           n²
     smaller        same        larger
        │            │            │
     Case 1        Case 2       Case 3
        │            │            │
       n          n log n        n²
```

---

## 9. Very Important: `log` Calculation

You'll frequently need:

$$
\log_b a
$$

For example:

$$
\log_2 8=3
$$

because:

$$
2^3=8
$$

So:

$$
n^{\log_2 8}=n^3
$$

Another example:

$$
\log_4 16=2
$$

because:

$$
4^2=16
$$

Therefore:

$$
n^{\log_4 16}=n^2
$$

---

## 10. Your Master Theorem Solving Procedure

Whenever you see a recurrence, use these **five steps**:

### Step 1

Put it into:

$$
T(n)=aT(n/b)+f(n)
$$

### Step 2

Identify:

$$
a,\quad b,\quad f(n)
$$

### Step 3

Calculate:

$$
\boxed{n^{\log_b a}}
$$

### Step 4

Compare:

$$
f(n)
$$

with:

$$
n^{\log_b a}
$$

### Step 5

Choose:

```text
f(n) smaller → Case 1 → Θ(n^log_b(a))

f(n) same    → Case 2 → Θ(n^log_b(a) log n)

f(n) larger  → Case 3 → Θ(f(n))
```

---

## One Important Limitation

Master Theorem **doesn't work for every recurrence**.

For example:

$$
T(n)=T(n-1)+1
$$

doesn't fit:

$$
T(n)=aT(n/b)+f(n)
$$

because `n - 1` is not `n/b`.

So for:

$$
T(n)=T(n-1)+1
$$

we need another technique, such as **expansion, recursion trees, or substitution**.

---

## Let's Test Whether the Concept Clicked

Don't solve a complicated problem yet.

Which **case** would this be?

$$
T(n)=4T(n/2)+n
$$

First calculate:

$$
n^{\log_2 4}
$$

Then compare it with:

$$
f(n)=n
$$

**Just give me your calculation and which case you think it is.**
