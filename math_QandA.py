"""
Dot product
-------------------------------------------------
[E] What’s the geometric interpretation of the dot product of two vectors?

"The dot product measures how much one vector points in the direction of another.
Geometrically, it equals the product of their magnitudes multiplied by the cosine of the angle between them, which is equivalent to the length of one vector times the projection of the other onto it.
A positive value indicates similar directions, zero indicates perpendicular vectors, and a negative value indicates opposite directions.
In machine learning, this interpretation is used to measure similarity between embeddings and forms the basis of transformer attention through query-key dot products."
-------------------------------------------------

-------------------------------------------------
[E] Given a vector  u
 , find vector  v
  of unit length such that the dot product of  u
  and  v
  is maximum.

Since v is constrained to have unit length, the dot product is u⋅v=∥u∥cosθ.
The magnitude of u is fixed, so the dot product is maximized when cosθ=1, i.e., when the angle between the vectors is 0
Therefore, v must point in the same direction as u, so the required vector is

v=u/|u|
and the maximum dot product is
|u|


-------------------------------------------------
Outer product
[E] Given two vectors  a=[3,2,1]
  and  b=[−1,0,1]
 . Calculate the outer product  aTb?


Example: Computing the Outer Product

Given two vectors:

    a = [3, 2, 1]
    b = [-1, 0, 1]

Step 1: Convert them into the correct shapes

Treat `a` as a column vector (3×1):

        [3]
    a = [2]
        [1]

Treat `b` as a row vector (1×3):

    bᵀ = [-1   0   1]

The outer product is:

          [3]
    abᵀ = [2] × [-1   0   1]
          [1]


Step 2: Compute each row

First row:
Multiply 3 by every element of b:

    3 × [-1, 0, 1]
    = [-3, 0, 3]

Second row:
Multiply 2 by every element of b:

    2 × [-1, 0, 1]
    = [-2, 0, 2]

Third row:
Multiply 1 by every element of b:

    1 × [-1, 0, 1]
    = [-1, 0, 1]


Step 3: Combine the rows

         [-3   0   3]
abᵀ =    [-2   0   2]
         [-1   0   1]
-------------------------------------------------

-------------------------------------------------
[M] Give an example of how the outer product can be useful in ML.

The outer product is widely used in neural network backpropagation to compute gradients of weight matrices.
If the input vector has shape n and the output error vector has shape m, their outer product produces an m×n gradient matrix that matches the shape of the weights.
It's also used to compute covariance matrices in PCA and statistics
-------------------------------------------------

-------------------------------------------------
[E] What does it mean for two vectors to be linearly independent?

Two vectors are linearly independent if neither vector can be expressed as a scalar multiple of the other.
Geometrically, they point in different directions and each contributes new information.
If one vector is simply a scaled version of the other, they are linearly dependent.
-------------------------------------------------


-------------------------------------------------
[M] Given two sets of vectors  A=a1,a2,a3,...,an
  and  B=b1,b2,b3,...,bm
 . How do you check that they share the same basis?

 INTERVIEW ANSWER
----------------
Two sets of vectors share the same basis (more precisely, they span the same vector space)
if they generate exactly the same set of vectors.

There are two common ways to verify this:

1. Span Test (Conceptual)
   - Every vector in A can be expressed as a linear combination of vectors in B.
   - Every vector in B can be expressed as a linear combination of vectors in A.
   If both conditions hold, then:
       span(A) = span(B)

2. Rank Test (Practical / Interview Preferred)
   - Form matrices A and B using the vectors as columns (or rows, consistently).
   - Compute:
         rank(A)
         rank(B)
         rank([A B])   # Concatenate the matrices
   If
         rank(A) == rank(B) == rank([A B])
   then both sets span the same vector space.

Quick Reminder
--------------
Think of each vector set as a toolbox.

If Toolbox A can build everything Toolbox B can build,
and Toolbox B can build everything Toolbox A can build,
then they span the same space.

Example
-------
A = {(1,0), (0,1)}

B = {(1,1), (1,-1)}

Although the vectors are different:

    (1,0) = 0.5*(1,1) + 0.5*(1,-1)
    (0,1) = 0.5*(1,1) - 0.5*(1,-1)

and

    (1,1)  = (1,0) + (0,1)
    (1,-1) = (1,0) - (0,1)

Each set can generate the other.

Therefore,

    span(A) = span(B)

and they are bases for the same vector space.

One-Line Interview Answer

"Two vector sets share the same basis (equivalently, span the same vector space)
if they generate exactly the same vectors. In practice, I verify this by checking
that rank(A) = rank(B) = rank([A B]), or equivalently, that each set can be
expressed as linear combinations of the other."
"Rank is the number of linearly independent vectors, which is also the dimension of the vector space (span) generated by those vectors."
-------------------------------------------------

[M] Given  n
  vectors, each of  d
  dimensions. What is the dimension of their span?
Norms and metrics
[E] What's a norm? What is  L0,L1,L2,Lnorm
 ?
[M] How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?







LEVEL-AI QUESTIONS
------------------------------------------------------------------------------------
ML QUESTIONS:

1. How is self-attention not better than multi-head attention? What advantages does it have?
2. How do transformers resolve the issue of exploding or vanishing gradients in beam search?
3. What are Sentence Transformers?
4. Why is the positional embedding not trained in the Transformer?
5. Rotary positional embedding, how is it better than the rest?
6. What is given as input to the decoder from the encoder in a Transformer? What all tokens are given as output by the Decoder?
7. BeamSearch in Decoder.
8. Describe working with multiple GPUs?
9. Details of RAG - chunking, retrieval etc



DSA QUESTIONS:
1.  DP Medium
2. Finding smallest number in a array using binary search
3. Basic binary search question (not on leetcode).
4. Valid Anagrams & Palindromes
5. Longest Palindromic Substring / Subsequence:
6. String Matching Algorithms
7. Maximum Subarray Sum
8. Longest Substring Without Repeating Characters
9. Subarray Product Less Than K
10. N-ary Trees / Prefix Trees (Trie)
11. Tree Traversals:
12. Graph Traversal (BFS/DFS):
13. Top K Frequent Words
14. LRU (Least Recently Used) Cache:
15. Merge K sorted list
16. Edit Distance (Levenshtein Distance):
17. 0/1 Knapsack:
18. Coin change
------------------------------------------------------------------------------------
"""