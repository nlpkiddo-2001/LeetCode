import collections
from collections import Counter, defaultdict
import math

from numba.cuda.libdevice import fast_logf
from sympy import false


#1. two sum
def twoSum_unsorted(arr, target):
    hash_map = {}
    for index, num in enumerate(nums):
        curr = target - num
        if curr in hash_map:
            return [hash_map.get(curr), index]
        hash_map[num] = index
    return [-1, -1]

# nums = [3,4,5,6]
# target = 7
# print(twoSum_unsorted(nums, target))

# def twoSum_sorted(arr, target):
#     i, j = 0, 0
#     while i < len(arr) and j >= 0:
#         if arr[i] + arr[j] == target:
#             return (i, j)
#         elif arr[i] + arr[j] < target:
#             j -= 1
#         else:
#             i += 1
#     return (-1, -1)
#####################################################################################################################################################################################################################

#2 1.2 Longest Substring Without Repeating Characters
def longestSubString(text: str):
    start = 0
    max_len = 0
    seen = set()

    for end in range(len(text)):
        while text[end] in seen:
            seen.remove(text[start])
            start += 1
        seen.add(text[end])
        max_len = max(max_len, end-start + 1)
    return max_len

# print(longestSubString("pwwkew"))
#####################################################################################################################################################################################################################
from typing import List, Optional, Any


#3 1.3 Merge Intervals Problem: Given a list of intervals, merge all overlapping intervals and return the result sorted by start time.
def mergeIntervals(big_list:List[List[int]]):
    big_list.sort(key=lambda x : x[0])

    result_list = []
    for interval in big_list:
        if not result_list:
            result_list.append(interval)

        if interval[0] <= result_list[-1][1]:
            result_list[-1][1] = max(interval[1], result_list[-1][1])
        else:
            result_list.append(interval)

    return result_list
#####################################################################################################################################################################################################################
from typing import List

class Node:
    def __init__(self, val: int):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1

        curr = self.head
        for _ in range(index):
            curr = curr.next

        return curr.val

    def insertHead(self, val: int):
        new_node = Node(val)

        new_node.next = self.head
        self.head = new_node

        if self.size == 0:
            self.tail = new_node

        self.size += 1

    def insertTail(self, val: int):
        new_node = Node(val)

        if self.size == 0:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

        self.size += 1

    def remove(self, index: int):
        if index < 0 or index >= self.size:
            return False

        if index == 0:
            self.head = self.head.next
            self.size -= 1

            if self.size == 0:
                self.tail = None

            return True

        prev = self.head
        for _ in range(index - 1):
            prev = prev.next

        to_delete = prev.next
        prev.next = to_delete.next

        if to_delete == self.tail:
            self.tail = prev

        self.size -= 1
        return True

    def getValues(self):
        values = []
        curr = self.head
        while curr:
            values.append(curr.val)
            curr = curr.next

        return values

    def getLen(self):
        return self.size

#####################################################################################################################################################################################################################
class DynamicArray:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.arr = [0] * capacity

    def get(self, i: int) -> int:
        return self.arr[i]

    def set(self, i: int, n: int) -> None:
        self.arr[i] = n

    def pushback(self, n: int) -> None:
        if self.size == self.capacity:
            self.resize()
        self.arr[self.size] = n
        self.size += 1

    def popback(self) -> int:
        if self.size == 0:
            raise IndexError("Array is Empty")

        val = self.arr[self.size - 1]
        self.size -= 1
        return val

    def resize(self) -> None:
        new_capacity = self.capacity * 2
        new_arr = [0] * new_capacity

        for i in range(self.size):
            new_arr[i] = self.arr[i]

        self.arr = new_arr
        self.capacity = new_capacity

    def getSize(self) -> int:
        return self.size

    def getCapacity(self) -> int:
        return self.capacity
#####################################################################################################################################################################################################################
from typing import List

class Node:
    def __init__(self, val: int):
        self.next = None
        self.prev = None
        self.val = val

class Deque:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def isEmpty(self):
        return self.size == 0

    def pushFront(self, val):
        new_node = Node(val)

        if self.isEmpty():
            self.head = self.tail = new_node

        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node

        self.size += 1

    def pushBack(self, val):
        new_node = Node(val)

        if self.isEmpty():
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

        self.size += 1

    def popFront(self):
        if self.isEmpty():
            return -1

        val = self.head.val
        self.head = self.head.next

        self.size -= 1

        if self.size == 0:
            self.tail = None
        else:
            self.head.prev = None

        return val

    def popBack(self):
        if self.isEmpty():
            return -1

        val = self.tail.val
        self.tail = self.tail.prev

        self.size -= 1

        if self.size == 0:
            self.head = None
        else:
            self.tail.next = None
        return val

    def getFront(self):
        return -1 if self.isEmpty() else self.head.val

    def getBack(self):
        return -1 if self.isEmpty() else self.tail.next

    def getValues(self):
        values = []
        curr = self.head

        while curr:
            values.append(curr.val)
            curr = curr.next

        return values


#####################################################################################################################################################################################################################
def hasDuplicate(nums: List[int]) -> bool:

    res = set()
    for num in nums:
        if num in res:
            return True
        else:
            res.add(num)
    return False

#####################################################################################################################################################################################################################

def isAnagram(s: str, t: str):
    hashMap1 = {}
    hashMap2 = {}

    for char in s:
        if char not in hashMap1:
            hashMap1[char] = 1
        else:
            value_count = hashMap1.get(char)
            hashMap1[char] = value_count + 1

    for char in t:
        if char not in hashMap2:
            hashMap2[char] = 1
        else:
            value_count = hashMap2.get(char)
            hashMap2[char] = value_count + 1

    hashMap1_keys = sorted(list(hashMap1.keys()))
    hashMap2_keys = sorted(list(hashMap2.keys()))

    hashMap1 = {i : hashMap1[i] for i in hashMap1_keys}
    hashMap2 = {i : hashMap2[i] for i in hashMap2_keys}

    if hashMap1 == hashMap2:
        return True
    else:
        return False

s = "racecar"
t = "carrace"
#
# if __name__ == "__main__":
#     print(isAnagram(s,t))
#####################################################################################################################################################################################################################

def groupAnagrams(strs):
    groups = {}

    for word in strs:
        sorted_word = "".join(sorted(word))

        if sorted_word not in groups:
            groups[sorted_word] = []

        groups[sorted_word].append(word)

    return list(groups.values())

# print(groupAnagrams(["act","pots","tops","cat","stop","hat"]))
#####################################################################################################################################################################################################################
def topKFrequent(nums: List[int], k: int) -> List[int]:
    freq_map = {}
    for num in nums:
        if num not in freq_map:
            freq_map[num] = 1
        else:
            freq_count = freq_map.get(num)
            freq_map[num] = freq_count + 1

    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq_map.items():
        buckets[count].append(num)

    results = []
    for count in range(len(nums), -1, -1):
        for num in buckets[count]:
            results.append(num)
            if len(results) == k:
                return results

nums = [1,2,2,2,3,3,3]
k = 2

# print(topKFrequent(nums, k))

#####################################################################################################################################################################################################################

class Solution:
    def encode(self, strs : List[str]):
        encoded = ""
        for s in strs:
            encoded += str(len(s)) + "#" + s
        return encoded

    def decode(self, strs : str):
        result = []
        i = 0
        while i < len(strs):
            j = i
            while s[j] != "#":
                j += 1
            j += 1
            length = int(s[i:j])
            sub_string = s[j : j + length]
            result.append(sub_string)
            i = j + length

#####################################################################################################################################################################################################################
# def productExceptSelf(nums: List[int]) -> List[int]:
#     n = len(nums)
#
#     out = [1] * n
#
#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 out[i] *= nums[j]
#
#     return out

def productExceptSelf(nums: List[int]) -> List[int]:
    n = len(nums)
    out = [1] * n


    prefix = 1
    for i in range(n):
        out[i] = prefix
        prefix *= nums[i]


    suffix = 1
    for i in range(n - 1, -1, -1):
        out[i] *= suffix
        suffix *= nums[i]

    return out

"""
dry run

out in prefix = [1, 1, 1, 1]
prefix = 1

i = 0
out = [1,1,1,1]
prefix = 1

i = 1
out = [1,1,1,1]
prefix = 2

i = 2
out = [1,1,2,1]
prefix = 8

i = 3
out = [1, 1, 2, 8]
prefix = 48


suffix 
i = 3
out = [1,1,2,8]
suffix = 6

i = 2
out = [1, 1, 12, 8]
suffix = 24

i = 1
out = [1, 24, 12, 8]
suffix = 48

i = 0
out = [48, 24, 12, 8]

"""
nums = [1, 2, 4, 6]
# print(productExceptSelf(nums))

#####################################################################################################################################################################################################################
def longestConsecutive(nums: List[int]) -> int:
    if not nums:
        return 0

    nums_set = set(nums)

    longest_streak = 0

    for num in nums_set:

        if (num - 1) not in nums_set:

            current_num = num
            current_streak = 1

            while (current_num + 1) in nums_set:
                current_num += 1
                current_streak += 1

            longest_streak = max(longest_streak, current_streak)
    return longest_streak

nums = [0,3,2,5,4,6,1,1]
# print(longestConsecutive(nums))
#####################################################################################################################################################################################################################
"""
Valid Palindrome
Given a string s, return true if it is a palindrome, otherwise return false.

A palindrome is a string that reads the same forward and backward. It is also case-insensitive and ignores all non-alphanumeric characters.
"""

import re
def isPalindrome(s: str) -> bool:
    s = re.sub(r'[^a-zA-Z0-9]', '', s)
    s = s.lower()
    n = len(s)
    i = 0
    j = n - 1
    while i < n and j > 0:
        if s[i] != s[j]:
            return False
        i += 1
        j -= 1

    return True

s = "Was it a car or a cat I saw?"
# print(isPalindrome(s))

#####################################################################################################################################################################################################################
"""
3Sum
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] where nums[i] + nums[j] + nums[k] == 0, and the indices i, j and k are all distinct.

The output should not contain any duplicate triplets. You may return the output and the triplets in any order.
"""
def threeSum(nums: List[int]):
    # res = set()
    # nums.sort()
    # for i in range(len(nums)):
    #     for j in range(i + 1, len(nums)):
    #         for k in range(j + 1, len(nums)):
    #             if nums[i] + nums[j] + nums[k] == 0:
    #                 tmp = [nums[i], nums[j], nums[k]]
    #                 res.add(tuple(tmp))
    # return [list(i) for i in res]

    res = []
    nums.sort()

    for i , a in enumerate(nums):
        if a > 0:
            break

        if i > 0 and a == nums[i - 1]:
            continue

        l , r = i + 1, len(nums) - 1

        while l < r:
            three_sum = a + nums[l] + nums[r]

            if three_sum > 0:
                r -= 1
            elif three_sum < 0:
                l += 1
            else:
                res.append([a, nums[l], nums[r]])
                l += 1
                r -= 1
                while nums[l] == nums[l - 1] and l < r:
                    l += 1

    return res

nums = [-1,0,1,2,-1,-4]
# print(threeSum(nums))

#####################################################################################################################################################################################################################
"""
Container With Most Water
You are given an integer array heights where heights[i] represents the height of the ith bar


You may choose any two bars to form a container. Return the maximum amount of water a container can store.
"""

def maxArea(heights: List[int]) -> int:
    l , r = 0, len(heights) - 1
    res = 0

    while l < r:
        area = min(heights[l], heights[r]) * (r - l)
        res = max(res, area)
        if heights[l] <= heights[r]:
            l += 1
        else:
            r -= 1
    return res

height = [1,7,2,5,4,7,3,6]
# print(maxArea(height))
#####################################################################################################################################################################################################################
"""
Best Time to Buy and Sell Stock
You are given an integer array prices where prices[i] is the price of NeetCoin on the ith day.

You may choose a single day to buy one NeetCoin and choose a different day in the future to sell it.


"""
def maxProfit(prices: List[int]) -> int:
    # brute force
    # res = 0
    # for i in range(len(prices)):
    #     buy = prices[i]
    #     for j in range(i + 1, len(prices)):
    #         sell = prices[j]
    #         res = max(res, sell - buy)
    # return res

    # two pointers
    left, right = 0, 1
    res = 0
    while right < len(prices):
        if prices[left] < prices[right]:
            curr_profit = prices[right] - prices[left]
            res = max(curr_profit, res)
        else:
            left = right
        right += 1

    return res

prices = [10,1,5,6,7,1]
# print(maxProfit(prices))
#####################################################################################################################################################################################################################
def longest_sub_string(strs: str) -> int:
    res = 0
    start = 0
    seen = set()

    for end in range(len(str)):
        while strs[end] in seen:
            seen.remove(strs[start])
            start+=1
        seen.add(strs[end])
        res = max(res, end-start + 1)
    return res

#####################################################################################################################################################################################################################
"""
Longest Repeating Character Replacement
You are given a string s consisting of only uppercase english characters and an integer k. You can choose up to k characters of the string and replace them with any other uppercase English character.

After performing at most k replacements, return the length of the longest substring which contains only one distinct character.
"""
def characterReplacement(s: str, k: int) -> int:
    count = {}
    res = 0

    l = 0

    for r in range(len(s)):
        count[s[r]] = 1 + count.get(s[r] , 0)

        while (r - l + 1) - max(count.values()) > k:
            count[s[l]] -= 1
            l += 1
        res = max( r - l + 1, res)

    return res

s = "AAABABB"
k = 1

print(characterReplacement(s, k))
#####################################################################################################################################################################################################################
"""
Minimum Window Substring
Given two strings s and t, return the shortest substring of s such that every character in t, including duplicates, is present in the substring. If such a substring does not exist, return an empty string "".

You may assume that the correct output is always unique.

Example 1:

Input: s = "OUZODYXAZV", t = "XYZ"

Output: "YXAZ"
"""
def minWindow(s: str, t: str) -> str:
    if t == "":
        return ""

    countT , window = {} , {}

    res, resLen = [-1, -1], float("infinity")

    l = 0

    for c in t:
        countT[c] = 1 + countT.get(c, 0)

    have, need = 0, len(countT)

    for r in range(len(s)):
        c = s[r]
        window[c] = 1 + window.get(c, 0)

        if c in countT and window[c] == countT[c]:
            have += 1

        while have == need:
            if (r - l + 1) < resLen:
                res = [l , r]
                resLen = (r - l + 1)

            window[s[l]] = window.get(s[l], 0) - 1

            if s[l] in countT and window[s[l]] < countT[s[l]]:
                have -= 1

            l += 1

    l, r = res
    return s[l : r + 1] if resLen != float('infinity') else ""


#####################################################################################################################################################################################################################
def isValid(s: str) -> bool:
    stack = []
    closeToOpen = {"}" : "{",  ")" : "(",  "]" : "["}

    for c in s:
        if c in closeToOpen:
            if stack and stack[-1] == closeToOpen[c]:
                stack.pop()
            else:
                return False
        else:
            stack.append(c)

    return True if not stack else False

#####################################################################################################################################################################################################################

google_dsa_questions = [

    "Two Sum / Four Sum",
    "Maximum Subarray Sum (Kadane's)",
    "Longest Substring Without Repeating Characters",
    "Longest Palindromic Substring",
    "Merge Intervals / Insert Interval",
    "Product of Array Except Self",
    "Set Matrix Zeroes / Rotate Matrix/Image",
    "Valid Parentheses (Stack)",
    "Search in Rotated Sorted Array (Binary Search)",
    "Sliding Window Maximum",

    "Reverse a Linked List (Iterative & Recursive)",
    "Detect Cycle in a Linked List (Floyd's Algorithm)",
    "Find the starting point of the Loop",
    "Merge Two Sorted Lists",
    "Remove Nth Node From End of List",
    "Clone a Linked List with Random Pointer",

    "Lowest Common Ancestor (LCA)",
    "Binary Tree Traversals (BFS/DFS)",
    "Invert/Flip Binary Tree",

# [Image of Inverted Binary Tree]

    "Binary Tree Maximum Path Sum / Diameter of Binary Tree",
    "Validate Binary Search Tree (BST)",
    "Serialize and Deserialize Binary Tree",
    "Number of Islands (BFS/DFS)",
    "Clone Graph",
    "Word Ladder (BFS)",
    "Detect Cycles in Graph (Directed/Undirected)",
    "Topological Sorting",

    "Longest Common Subsequence (LCS)",
    "Coin Change Problem",
    "Longest Increasing Subsequence",
    "Word Break Problem",
    "Generate All Possible Balanced Parentheses (Backtracking)",
    "Permutations and Combinations (Backtracking)",
    "N-Queens Problem (Backtracking)",
    "Sudoku Solver (Backtracking)",

    "LRU Cache (HashMap + Doubly Linked List)",
    "LFU Cache",
    "Top K Frequent Elements (Heap/Priority Queue)",
    "Find Median from Data Stream (Two Heaps)"
]
#####################################################################################################################################################################################################################

def four_sum_two_pointers(nums: List[int], target: int):
    n = len(nums)

    if n < 4:
        return []
    nums.sort()

    result = []

    for i in range(n - 3):

        if i > 0 and nums[i] == nums[i-1]:
            continue

        for j in range(i + 1, n-2):

            if j > i + 1 and nums[j] == nums[j-1]:
                continue

            remaining_target = target - nums[i] - nums[j]

            left = j + 1
            right = n - 1

            while left < right:

                current_sum = nums[left] + nums[right]

                if current_sum == remaining_target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])


                    while left < right and nums[left] == nums[left + 1]:
                        left += 1

                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1
                    right -= 1

                elif current_sum < remaining_target:
                    left += 1

                else:
                    right -= 1

        return result
#####################################################################################################################################################################################################################
def max_subarray_sum_kadanes(nums: list[int]) -> int:
    if not nums:
        return 0

    current_max = nums[0]
    global_max = nums[0]

    for i in range(1, len(nums)):
        num = nums[i]

        current_max = max(num, current_max + num)
        global_max = max(global_max, current_max)

    return global_max
    

array1 = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
result1 = max_subarray_sum_kadanes(array1)
# print(f"Array: {array1} -> Max Sum: {result1}")


def longest_sub_string(text: str):
    start = 0
    max_len = 0
    seen = set()

    for end in range(len(text)):
        while seen[end] in text:
            seen.remove(text[start])
            start += 1
        seen.add(text[end])
        max_len = max(max_len, end-start+1)
    return max_len



def longest_palindrome(s: str):
    if not s:
        return ""

    start, end = 0, 0

    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left+1 , right - 1

    for i in range(len(s)):
        l1, r1 = expand(i, i)
        if (r1 - l1) > (end - start):
            start , end = l1, r1

        l2, r2 = expand(i, i+1)
        if (r2 - l2) > (end - start):
            start, end = l2, r2

    return s[start: end+1]

def merge_intervals(big_list: List[List[int]]):
    result_list = []

    big_list.sort(lambda x : x[0])

    for interval in big_list:
        if not result_list:
            result_list.append(interval)


        if interval[0] <= result_list[-1][1]:
            result_list[-1][1] = max(interval[1], result_list[-1][1])
        else:
            result_list.append(interval)

    return result_list

def product_except_self(nums: List[int]):
    n = len(nums)
    out = [1] * n

    prefix = 1
    for i in range(n):
        out[i] = prefix
        prefix *= nums[i]

    suffix = 1
    for i in range(n-1, -1, -1):
        out[i] *= suffix
        suffix *= nums[i]

    return out


def setZeros(matrix: List[List[int]]) -> None:
    rows, cols = len(matrix), len(matrix[0])
    firstRowZero = False
    firstColZero = False

    # check first row
    for c in range(cols):
        if matrix[0][c] == 0:
            firstRowZero = True
            break


    # check first col
    for r in range(rows):
        if matrix[r][0] == 0:
            firstColZero = True
            break

    # mark rows and cols
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 0:
                matrix[0][c] = 0
                matrix[r][0] = 0


    # apply mark
    for r in range(rows):
        for c in range(cols):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0

    # fix first row
    if firstRowZero:
        for c in range(cols):
            matrix[0][c] = 0

    # fix first col
    if firstColZero:
        for r in range(rows):
            matrix[r][0] = 0


def rotate_matrix_by_90_degree(matrix: List[List[int]]):
    l , r = 0, len(matrix) - 1

    while l < r:
        for i in range(r - l):
            top, bottom = l , r

            #save top left
            topLeft = matrix[top][l + i]

            # move bottom left into top left
            matrix[top][l + 1] = matrix[bottom - i][l]

            # move bottom right into bottom left
            matrix[bottom - i][l] = matrix[bottom][r - i]

            # move top right into bottom right
            matrix[bottom][r - i] = matrix[top + i][r]

            # move top left into top right
            matrix[top + i][r] = topLeft

        r-=1
        l+=1



def isValidParentheses(s: str):
    stack = []
    closeToOpen = {
        "}" : "{",
        ")" : "(",
        "]" : "["
    }

    for c in s:
        if c in closeToOpen:
            if stack and stack[-1] == closeToOpen[c]:
                stack.pop()
            else:
                return False
        else:
            stack.append(c)

    return True if not stack else False


def search_rotated_sorted_array(nums: List[int], target: int):
    l , r = 0 , len(nums) - 1

    while l <= r:
        mid = (l + r) // 2

        if nums[mid] == target:
            return mid


        if nums[l] <= nums[mid]:
            if target > nums[mid] or target < nums[l]:
                l = mid + 1
            else:
                r = mid - 1

        else:
            if target < nums[mid] or target > nums[r]:
                r = mid - 1
            else:
                l = mid + 1

    return -1

def maxSlidingWindow(nums: List[int], k: int):
    l = r = 0
    output = []
    q = collections.deque()

    while r < len(nums):
        while q and nums[q[-1]] < nums[r]:
            q.pop()
        q.append(r)

        if l > q[0]:
            q.popleft()

        if (r + 1) >= k:
            output.append(q[0])
            l += 1

        r += 1

    return output




"""
Example Google machine learning engineer interview questions: Coding

Graphs / Trees (39% of questions, most frequent)
"Given a binary tree, find the maximum path sum. The path may start and end at any node in the tree." (Solution)
"Given an encoded string, return its decoded string." (Solution)
"Given two words (beginWord and endWord), and a dictionary's word list, find the length of the shortest transformation sequence from beginWord to endWord, such that: 1) Only one letter can be changed at a time, and 2) Each transformed word must exist in the word list." (Solution)
"Given a matrix of N rows and M columns. From m[i][j], we can move to m[i+1][j], if m[i+1][j] > m[i][j], or can move to m[i][j+1] if m[i][j+1] > m[i][j]. The task is print longest path length if we start from (0, 0)." (Solution)
Arrays / Strings (26%)
Implement a SnapshotArray that supports pre-defined interfaces (note: see link for more details). (Solution)
"In a row of dominoes, A[i] and B[i] represent the top and bottom halves of the i-th domino.  (A domino is a tile with two numbers from 1 to 6 - one on each half of the tile.) We may rotate the i-th domino, so that A[i] and B[i] swap values. Return the minimum number of rotations so that all the values in A are the same, or all the values in B are the same. If it cannot be done, return -1." (Solution)
"Your friend is typing his name into a keyboard.  Sometimes, when typing a character c, the key might get long pressed, and the character will be typed 1 or more times. You examine the typed characters of the keyboard.  Return True if it is possible that it was your friend's name, with some characters (possibly none) being long pressed." (Solution)
"Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n)." (Solution)
"Given a list of query words, return the number of words that are stretchy." Note: see link for more details. (Solution)
Dynamic Programming (12%)
"Given a matrix and a target, return the number of non-empty submatrices that sum to target." (Solution)
"Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area." (Solution)
"Your car starts at position 0 and speed +1 on an infinite number line. (Your car can go into negative positions.) Your car drives automatically according to a sequence of instructions A (accelerate) and R (reverse)...Now for some target position, say the length of the shortest sequence of instructions to get there." (Solution)
Recursion (12%)
"A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down). Find all strobogrammatic numbers that are of length = n." (Solution)
"Given a binary tree, find the length of the longest path where each node in the path has the same value. This path may or may not pass through the root. The length of path between two nodes is represented by the number of edges between them." (Solution)
Geometry / Math (11% of questions, least frequent)
"You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contains a single digit. Add the two numbers and return it as a linked list." (Solution)

"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BinarySearchTree:
    def maxSumPath(self, root: TreeNode):

        self.max_sum = float("-inf")

        def maxSum(node):
            if not node:
                return 0

            left_gain = max(maxSum(node.left), 0)
            right_gain = max(maxSum(node.right), 0)

            current_sum = node.val + left_gain + right_gain

            self.max_sum = max(self.max_sum, current_sum)

            return node.val + max(left_gain, right_gain)

        maxSum(root)
        return self.max_sum

def create_test_tree():
    """Creates the following binary tree:
               10
              /  \
             2   10
            / \     \
           20   1  -25
                     /  \
                    3    4
    Expected max path sum: 42 (20 -> 2 -> 10 -> 10)
    """
    root = TreeNode(10)
    root.left = TreeNode(2)
    root.right = TreeNode(10)

    root.left.left = TreeNode(20)
    root.left.right = TreeNode(1)

    root.right.right = TreeNode(-25)
    root.right.right.left = TreeNode(3)
    root.right.right.right = TreeNode(4)

    return root


# Main testing block
if __name__ == "__main__":
    # Create an instance of BinarySearchTree
    tree = BinarySearchTree()

    # Build the test tree
    root = create_test_tree()

    # Call the maxSumPath method and display the result
    result = tree.maxSumPath(root)
    # print("Maximum Path Sum:", result)



def decodeString(s: str):
    stack = []
    current_string = ""
    k = 0

    for char in s:
        if char.isdigit():
            k = k * 10 + int(char)

        elif char == "[":
            stack.append((current_string, k))

            current_string = ""
            k = 0

        elif char == "]":
            previous_string, count  = stack.pop()

            current_string = previous_string + current_string * count

        else:
            current_string += char

    return current_string




s = "3[a]2[bc]"

# print(decodeString(s))



def decodeStringWithoutSquareBrackets(s: str):
    result = ""
    i = 0

    while i < len(s):
        k = 0
        while i < len(s) and s[i].isdigit():
            k = k * 10 + int(s[i])
            i += 1


        encoded_string = ""
        while i < len(s) and s[i].isalpha():
            encoded_string += s[i]
            i += 1

        result += encoded_string * k

    return result


def findMin(nums: List[int]) -> int:
    l, r = 0, len(nums) - 1

    while l < r:
        mid = (l + r) // 2

        if nums[mid] > nums[r]:
            l = mid + 1

        else:
            r = mid

    return nums[l]

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    prev, curr = None, head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev

def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy

    while list1 and list2:
        if list1.val < list2.val:
            tail.next = list1.val
            list1 = list1.next

        else:
            tail.next = list2.val
            list2 = list2.next
        tail = tail.next

    if list1:
        tail.next = list1
    if list2:
        tail.next = list2

    return dummy.next

def hasCycle(head: Optional[ListNode]) -> bool:
    slow, fast = head, head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def reorderList(head: Optional[ListNode]) -> None:
    # find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # reverse second half
    prev = None
    curr = slow.next
    slow.next = None

    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt

    second = prev
    first = head

    # link
    while second:
        temp1 = first.next
        temp2 = second.next

        first.next = second
        second.next = temp1

        first = temp1
        second = temp2

def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    slow = fast = dummy

    # move fast n steps ahead
    for _ in range(n):
        fast = fast.next

    # iterate fast till end
    while fast.next:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next

    return dummy.next

import heapq

def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    heap = []
    counter = 0

    for node in lists:
        if node:
            heapq.heappush(heap, (node.val, counter, node))
            counter += 1

    dummy = ListNode(0)
    tail = dummy

    while heap:

        val, _, node = heapq.heappop(heap)

        tail.next = node
        tail = tail.next

        if node.next:
            heapq.heappush(heap, (node.next.val, counter, node))
            counter += 1

    return dummy.next




def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    heap = []
    counter = 0

    for node in lists:
        if node:
            heapq.heappush(heap, (node.val, counter, node))
            counter += 1


    dummy = ListNode(0)
    tail = dummy

    while heap:
        val, _, node = heapq.heappop(heap)

        tail.next = node
        tail = tail.next

        if node.next:
            heapq.heappush(heap, (node.next.val, counter, node.next))
            counter += 1

    return dummy.next

from collections import deque
from typing import Optional, List, Any
import collections


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def invertTree(self, root):
        """
        PATTERN: "Mirror Swap" - Swap left and right at every node
        MEMORY AID: "Like looking in a mirror - everything flips"

        Example:
        Input:     4           Output:    4
                  / \                    / \
                 2   7                  7   2
                / \ / \                / \ / \
               1 3 6 9              9 6 3 1

        Keywords: swap, mirror, recursive, BFS/DFS both work
        """
        if not root:
            return None

        root.left, root.right = root.right, root.left

        self.invertTree(root.left)
        self.invertTree(root.right)

        return root

    def invertTreeIteratively(self, root):
        """
        PATTERN: "Level by Level Mirror" - Use queue to swap each level
        """
        if root is None:
            return None

        queue = deque([root])

        while queue:
            node = queue.popleft()

            node.left, node.right = node.right, node.left

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return root




def maxDepth(root: Optional[TreeNode]) -> int:
    """
    PATTERN: "Depth Counting" - Count levels going down
    MEMORY AID: "How many floors in this building?"

    Example:
    Input:    3           Output: 3
             / \
            9  20
              /  \
             15   7

    Levels: Root(3) -> Level1(9,20) -> Level2(15,7) = 3 levels
    Keywords: recursive, max of left/right + 1, base case empty = 0
    """
    if not root:
        return 0

    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)

    return 1 + max(left_depth, right_depth)

def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    PATTERN: "Twin Checker" - Compare structure AND values
    MEMORY AID: "Are these twins identical in every way?"

    Example:
    Tree1:  1        Tree2:  1        Result: True
           / \              / \
          2   3            2   3

    Tree1:  1        Tree2:  1        Result: False
           /                \
          2                  2

    Keywords: both null=true, one null=false, values match + recursive check
    """
    if p is None and q is None:
        return True

    if p is None or q is None:
        return False

    if p.val != q.val:
        return False

    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)




def isSubTree(root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    """
    PATTERN: "Find the Mini-Me" - Look for exact subtree match anywhere
    MEMORY AID: "Is this small tree hiding somewhere in the big tree?"

    Example:
    Main Tree:    3          SubTree:  4         Result: True
                 / \                   / \
                4   5                 1   2
               / \
              1   2

    Keywords: check each node as potential root, use isSameTree helper
    """
    if root is None:
        return False

    if isSameTree(root, subRoot):
        return True

    return isSubTree(root.left, subRoot) or isSubTree(root.right, subRoot)


def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    PATTERN: "BST Split Point" - Use BST property to navigate
    MEMORY AID: "Where do the paths to p and q first split?"

    Example: BST with nodes p=2, q=8
         6
        / \
       2   8
      / \ / \
     0 4 7 9
        / \
       3   5

    LCA = 6 (first node where paths split: 6->2 vs 6->8)
    Keywords: BST property, both left/both right/split point
    """
    curr = root

    while curr:
        if p.val < curr.val and q.val < curr.val:
            curr = curr.left

        elif p.val > curr.val and q.val > curr.val:
            curr = curr.right

        else:
            return curr


def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    """
    PATTERN: "Floor by Floor" - Process each level separately
    MEMORY AID: "Print each floor of the building separately"

    Example:
    Input:    3           Output: [[3], [9,20], [15,7]]
             / \
            9  20
              /  \
             15   7

    Level 0: [3]
    Level 1: [9, 20]
    Level 2: [15, 7]

    Keywords: BFS, queue, track level size, process level at once
    """
    res = []
    q = collections.deque()
    q.append(root)

    while q:
        qLen = len(q)
        level = []

        for i in range(qLen):
            node = q.popleft()
            if node:
                level.append(node.val)
                q.append(node.left)
                q.append(node.right)
        if level:
            res.append(level)
    return res



def isValidBST(root: Optional[TreeNode]) -> bool:
    """
    PATTERN: "Range Validator" - Each node must be within min/max bounds
    MEMORY AID: "Every node has a valid range it must stay within"

    Example:
    Valid BST:    5           Invalid BST:    5
                 / \                         / \
                3   8                       3   8
               / \ / \                     / \ / \
              2 4 6 9                    2 6 4 9
                                           ^invalid: 6>5

    Node 5: range(-∞, ∞)
    Node 3: range(-∞, 5)
    Node 8: range(5, ∞)
    Node 6: range(5, 8) ✓

    Keywords: recursive bounds, left<node<right, update min/max bounds
    """

    def valid(node, left, right):
        if not node:
            return True
        if not (node.val > left and node.val < right):
            return False

        return valid(node.left, left, node.val) and valid(node.right, node.val, right)

    return valid(root, float("-inf"), float("inf"))



def kthSmallest(root: Optional[TreeNode], k: int) -> int:
    """
    PATTERN: "In-Order Counting" - BST in-order gives sorted sequence
    MEMORY AID: "Walk the tree in sorted order, count to k"

    Example: k=3
    BST:    3
           / \
          1   4
           \
            2

    In-order traversal: 1, 2, 3, 4
    k=3 means 3rd smallest = 3

    Keywords: in-order traversal, counter, stop when count==k
    """
    count = 0
    result = None

    def inOrder(node):
        nonlocal count, result

        if not node or result is not None:
            return

        inOrder(node.left)

        count += 1
        if count == k:
            result = node.val
            return

        inOrder(node.right)

    inOrder(root)
    return result



def buildTree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    PATTERN: "Preorder Root + Inorder Split" - Use preorder root to split inorder
    MEMORY AID: "Preorder tells me the root, inorder tells me left/right parts"

    Example:
    preorder = [3,9,20,15,7]
    inorder =  [9,3,15,20,7]

    Step 1: Root = 3 (first in preorder)
    Step 2: Split inorder at 3: [9] | 3 | [15,20,7]
    Step 3: Left subtree from [9], Right subtree from [20,15,7]

    Keywords: preorder[0]=root, find root in inorder, split arrays, recursive build
    """
    if not inorder or not preorder:
        return None

    root_val = preorder[0]
    root = TreeNode(root_val)

    mid = inorder.index(root_val)

    root.left = buildTree(preorder[1:mid + 1], inorder[:mid])
    root.right = buildTree(preorder[mid + 1:], inorder[mid + 1:])

    return root



def maxPathSum(root: Optional[TreeNode]) -> int:
    """
    PATTERN: "Path Through Node" - Consider each node as path center
    MEMORY AID: "What's the best path that goes through this node?"

    Example:
    Tree:    1           Max path: 2->1->3 = 6
            / \
           2   3

    At node 1: left_gain=2, right_gain=3, path_sum=2+1+3=6
    At node 2: left_gain=0, right_gain=0, path_sum=2
    At node 3: left_gain=0, right_gain=0, path_sum=3

    Keywords: DFS, max gain from subtrees, path through current node, global max
    """
    maxSum = float("-inf")

    def dfs(node: Optional[TreeNode]):
        nonlocal maxSum

        if not node:
            return 0

        left_gain = max(dfs(node.left), 0)
        right_gain = max(dfs(node.right), 0)

        current_sum = node.val + left_gain + right_gain

        maxSum = max(current_sum, maxSum)

        return node.val + max(left_gain, right_gain)

    dfs(root)
    return maxSum



class Codec:
    """
    PATTERN: "Preorder with Nulls" - Serialize as preorder, mark nulls explicitly
    MEMORY AID: "Write down the tree as you visit it, mark empty spots"

    Example:
    Tree:    1           Serialized: "1,2,#,#,3,4,#,#,5,#,#"
            / \
           2   3
              / \
             4   5

    Preorder: 1 -> 2 -> null -> null -> 3 -> 4 -> null -> null -> 5 -> null -> null
    Keywords: preorder traversal, "#" for null, recursive deserialize with index
    """

    def serialize(self, root: Optional[TreeNode]) -> str:
        result = []

        def dfs(node):
            if not node:
                result.append("#")
                return

            result.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ",".join(result)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        values = data.split(",")
        self.index = 0

        def dfs():
            if values[self.index] == "#":
                self.index += 1
                return None

            node = TreeNode(int(values[self.index]))
            self.index += 1
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()

import heapq

class MedianFinder:

    def __init__(self):
        self.left = []
        self.right = []

    def addNum(self, num: int):
        heapq.heappush(self.left, -num)

        if self.right and (-self.left[0] > self.right[0]):
            val = -heapq.heappop(self.left)
            heapq.heappush(self.right, val)

        if len(self.left) > len(self.right) + 1:
            val = heapq.heappop(self.left)
            heapq.heappush(self.right, val)
        elif len(self.right) > len(self.left):
            val = heapq.heappop(self.right)
            heapq.heappush(self.left, -val)

    def findMedian(self):
        if len(self.left) > len(self.right):
            return float(-self.left[0])
        else:
            return (-self.left[0] + self.right[0]) / 2.0



def combinationSum(nums, target):
    result = []

    def backtrack(start, remaining, path):
        if remaining == 0:
            result.append(path[:])
            return

        if remaining < 0:
            return

        for i in range(start, len(nums)):
            path.append(nums[i])

            needed = remaining - nums[i]

            backtrack(i, needed, path)

            path.pop()

    backtrack(0, target, [])
    return result

nums = [2,5,6,9]
target = 9
# print(combinationSum(nums, target))

def exist(board: List[List[str]], word: str) -> bool:
    rows, cols = len(board), len(board[0])

    def dfs(r, c, index):
        if index == len(word):
            return True

        if (r < 0 or r >= rows) or (c < 0 or c >= cols) or (board[r][c]!=word[index]):
            return False

        temp = board[r][c]
        board[r][c] = "#"

        found = (
            dfs(r + 1, c , index+1) or
            dfs(r - 1, c,  index+1) or
            dfs(r, c + 1, index + 1) or
            dfs(r, c - 1, index + 1)
        )

        board[r][c] = temp
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True

    return False


board = [
  ["A","B","C","D"],
  ["S","A","A","T"],
  ["A","C","A","E"]
]
word = "CAT"

# print(exist(board, word))



def reverse_Words(s):
    s = s[::-1]

    n = len(s)
    result = []
    i = 0

    l = 0
    while l < n:
        if s[l] != '.':

            if i != 0:
                result.append('.')
                i += 1

            r = l
            while r < n and s[r]!='.':
                result.append(s[r])
                i+=1
                r += 1

            result[i - (r - l):i] = reversed(result[i - (r - l):i])
            l = r

        l+=1
    return ''.join(result)



def sort012(arr):
    n = len(arr)

    lo = 0
    hi = n - 1
    mid = 0

    while mid <= hi:
        if arr[mid] == 0:
            arr[lo], arr[mid] = arr[mid], arr[lo]
            lo += 1
            mid += 1

        elif arr[mid] == 1:
            mid += 1

        else:
            arr[mid], arr[hi] = arr[hi], arr[mid]
            hi -= 1

    return arr


arr = [0, 1, 2, 0, 1, 2]
sort012(arr)

import heapq

def kth_Smallest(arr, k):
    pq = []

    for i in range(len(arr)):
        heapq.heappush(pq, -arr[i])

        if len(pq) > k:
            heapq.heappop(pq)

    return -pq[0]


def isPower(x, y):
    res1 = math.log(x) / math.log(y)
    return res1 == math.floor(res1)



def removeSpaces(input_string):
    list = []

    for i in range(len(input_string)):
        if input_string[i] != " ":
            list.append(input_string[i])

    return "".join(list)


def heapify(arr, n, i):
    largest = i

    l = 2 * i + 1

    r = 2 * i + 1

    if l < n and arr[l] < arr[largest]:
        largest = l

    if r < n and arr[r] < arr[largest]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]

        heapify(arr, n, largest)


def heapSort(arr):
    n = len(arr)

    for i in range(n // 2 - 1 , -1 , -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]

        heapify(arr, i, 0)

arr = [9, 4, 3, 8, 10, 2, 5]
heapSort(arr)

def bubbleSort(arr):
    n = len(arr)

    for i in range(n):
        swapped = False

        for j in range(0, n-i-1):

            if arr[j] > arr[j + 1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True

        if not swapped:
            break

bubbleSort(arr)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class PrefixTree:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def startsWith(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return False


class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        def dfs(node, index):
            if index == len(word):
                return node.is_end

            ch = word[index]

            if ch == ".":
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
                return False
            else:
                if ch not in node.children:
                    return False

                return dfs(node.children[ch], index + 1)

        return dfs(self.root, 0)

class TrieNodeDummy:
    def __init__(self):
        self.children = {}
        self.is_word = False

    def addWord(self, word: str):
        curr = self
        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = TrieNode()

            curr = curr.children[ch]
        curr.is_word = True


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNodeDummy()

        for word in words:
            root.addWord(word)

        ROWS, COLS = len(board), len(board[0])
        res, visit = set(), set()

        def dfs(r, c, node: TrieNodeDummy, word):

            if (r < 0 or
                c < 0 or
                r == ROWS or
                c == COLS or
                board[r][c] not in node.children or
                word in visit):
                return

            visit.add((r, c))

            node = node.children[board[r][c]]
            word += board[r][c]

            if node.is_word:
                res.add(word)

            dfs(r + 1, c, node, word)
            dfs(r - 1, c, node, word)
            dfs(r, c + 1, node, word)
            dfs(r, c - 1, node, word)

            visit.remove((r, c))

        for r in range(ROWS):
            for c in range(COLS):
                dfs(r, c, root, "")

        return list(res)

board = [
    ["o","a","a","n"],
    ["e","t","a","e"],
    ["i","h","k","r"],
    ["i","f","l","v"]
]

words = ["oath","pea","eat","rain"]

# res = Solution().findWords(board, words)


def numIslands(grid: List[List[str]]) -> int:
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    islands = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == "0":
            return

        grid[r][c] = "0"

        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                islands += 1
                dfs(r, c)

    return islands

grid = [
    ["0","1","1","1","0"],
    ["0","1","0","1","0"],
    ["1","1","0","0","0"],
    ["0","0","0","0","0"]
  ]

# print(numIslands(grid))

class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
    oldToNew = {}

    def dfs(root):
        if root in oldToNew:
            return oldToNew[root]

        copy = Node(root.val)
        oldToNew[root] = copy

        for nei in root.neighbors:
            copy.neighbors.append(dfs(nei))

        return copy

    return dfs(node) if not None else None

def pacificAtlantic(heights: List[List[int]]) -> List[List[int]]:
    ROWS, COLS = len(heights), len(heights[0])
    pac, atl = set(), set()

    def dfs(r, c, visit, prevHeight):
        if (r, c) in visit or r < 0 or c < 0 or r == ROWS or c == COLS or heights[r][c] < prevHeight:
            return

        visit.add((r, c))

        dfs(r + 1, c, visit, heights[r][c])
        dfs(r - 1, c, visit, heights[r][c])
        dfs(r, c + 1, visit, heights[r][c])
        dfs(r, c - 1, visit, heights[r][c])

    for c in range(COLS):
        dfs(0, c, pac, heights[0][c])
        dfs(ROWS - 1, c , atl, heights[ROWS - 1][c])

    for r in range(ROWS):
        dfs(r, 0, pac, heights[r][0])
        dfs(r, COLS-1, atl, heights[r][COLS-1])

    res = []
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in pac and (r, c) in atl:
                res.append([r, c])

    return res

def canFinish(numCourses, prerequisites):
    graph = {i : [] for i in range(numCourses)}

    for course, prequi in prerequisites:
        graph[course].append(prequi)

    state = [0] * numCourses

    def dfs(course):
        if state[course] == 1:
            return False
        if state[course] == 2:
            return True

        state[course] = 1

        for next_course in graph[course]:
            if not dfs(next_course):
                return False

        state[course] = 2

        return True

    for c in range(numCourses):
        if state[c] == 0:
            if not dfs(c):
                return False

    return True

def is_valid_tree(n, edges):
    if len(edges) != n - 1:
        return False

    graph = {i : [] for i in range(n)}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()

    def dfs(node, parent):
        if node in visited:
            return False

        visited.add(node)

        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            if not dfs(neighbor, node):
                return False

        return True

    if not dfs(0, -1):
        return False

    return len(visited) == n

def countComponents(n, edges):
    graph = {i : [] for i in range(n)}

    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    components = 0

    def dfs(node):
        visited.add(node)
        for nei in graph[node]:
            if nei not in visited:
                dfs(nei)

    for i in range(n):
        if i not in visited:
            dfs(i)
            components += 1
    return components


"""
Was asked the following question during my onsite. Ran out of time before forming a full solution, still not sure what a good approach to this would be.

Given on-call rotation schedule for multiple people by: their name, start time and end time of the rotation:

Abby 1 10
Ben 5 7
Carla 6 12
David 15 17

Your goal is to return rotation table without overlapping periods representing who is on call during that time. Return "Start time", "End time" and list of on-call people:

1 5 Abby
5 6 Abby, Ben
6 7 Abby, Ben, Carla
7 10 Abby, Carla
10 12 Carla
15 17 David
"""
from collections import defaultdict

def build_rotation_table(schedule):
    events = []

    for name, start, end in schedule:
        events.append((start, 1, name))
        events.append((end, -1, name))

    events.sort(key=lambda x:(x[0] , x[1]))

    active = set()
    result = []

    i = 0
    prev_time = None

    while i < len(events):
        time = events[i][0]

        if prev_time is not None and prev_time < time and active:
            result.append((prev_time, time, sorted(active)))

        while i < len(events) and events[i][0] == time:
            _, type, name = events[i]

            if type==1:
                active.add(name)
            else:
                active.remove(name)

            i += 1

        prev_time = time

    return result



schedule = [
    ("Abby", 1, 10),
    ("Ben", 5, 7),
    ("Carla", 6, 12),
    ("David", 15, 17)
]

for row in build_rotation_table(schedule):
    print(row)


"""
Alien Dictionary
Hard
Topics
Company Tags
Hints
There is a new alien language that uses the English alphabet, but the order of the letters is unknown.

You are given a list of strings words from the alien language's dictionary. It is claimed that the strings in words are sorted lexicographically by the rules of this new language.

If this claim is incorrect, and the given arrangement of strings in words cannot correspond to any order of letters, return "".

Otherwise, return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there are multiple solutions, return any of them.

A string a is lexicographically smaller than a string b if either of the following is true:

The first letter where they differ is smaller in a than in b.
a is a prefix of b and a.length < b.length.

Example 1:

Input: words = ["z","o"]

Output: "zo"
Explanation:
From "z" and "o", we know 'z' < 'o', so return "zo".


Example 2:

Input: words = ["hrn","hrf","er","enn","rfnn"]

Output: "hernf"
"""

from collections import deque

class Solution:
    def alienOrder(self, words):
        # creating graph and indegree
        graph = {}
        indegree = {}

        for word in words:
            for ch in word:
                graph[ch] = set()
                indegree[ch] = 0

        # building graph
        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i+1]

            # checking if it is a valid prefix else returning ""
            if len(word1) > len(word2) and word1.startswith(word2):
                return ""

            length = min(len(word1), len(word2))

            # finding first different character
            for j in range(length):
                if word1[j] != word2[j]:
                    parent = word1[j]
                    child = word2[j]

                    # Adding edge once
                    if child not in graph[parent]:
                        graph[parent].add(child)
                        indegree[child] += 1

                    break

        # queue indegree with 0
        queue = deque()

        for ch in indegree:
            if indegree[ch] == 0:
                queue.append(ch)


        # topological sorting
        answer = []

        while queue:
            node = queue.popleft()

            answer.append(node)

            for neighbor in graph[node]:
                indegree[neighbor] -= 1

                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        # cycle detection
        if len(answer) != len(graph):
            return ""

        return "".join(answer)


class Solution:
    def climbStairs(self, n:int) -> int:
        first = 1
        second = 2

        for i in range(3, n+1):
            current = first + second
            first = second
            second = current

        return second



class Solution:
    def rob(self, nums):
        if len(nums) == 1:
            return nums[0]

        prev2 = nums[0]
        prev1 = max(nums[0], nums[1])

        for i in range(2, len(nums)):
            current = max(prev1, nums[i] + prev2)
            prev2 = prev1
            prev1 = current

        return prev1


class Solution:
    def rob(self, nums):
        if len(nums) == 1:
            return nums[0]

        def rob_inner(arr):
            prev2 = 0
            prev1 = 0

            for money in arr:
                current = max(prev1, money + prev2)
                prev2 = prev1
                prev1 = current

            return prev1

        return max(
            rob_inner(nums[:-1]),
            rob_inner(nums[1:])
        )


class Solution:

    def longestPalindrome(self, s: str) -> str:
        start = 0
        end = 0

        def expand(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1

            return left + 1, right - 1

        for i in range(len(s)):
            l1, r1 = expand(i, i)

            if r1 - l1 > end - start:
                start, end = l1, r1

            l2, r2 = expand(i, i+1)

            if r2 - l2 > end - start:
                start, end = l2, r2

        return s[start: end+1]

class Solution:
    def countSubstrings(self, s: str) -> int:

        def expand(left, right):
            count = 0
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
            return count

        answer = 0
        for i in range(len(s)):
            answer += expand(i, i)

            answer += expand(i, i+1)

        return answer




class Solution1:

    @staticmethod
    def numDecodings(s: str) -> int:
        n = len(s)

        dp = [0] * (n + 1)
        dp[n] = 1

        for i in range(n-1, -1, -1):
            if s[i] == '0':
                dp[i] = 0
                continue

            dp[i] = dp[i + 1]

            if (
                i + 1 < n and
                    (
                        s[i] == '1' or
                        (s[i] == '2' and s[i] <= '6')
                    )
            ):
                dp[i] += dp[i + 2]

        return dp[0]

s = "1245"
print(Solution1.numDecodings(s))


def coinChange(coins, amount):
    INF = amount + 1

    dp = [INF] * (amount + 1)
    dp[0] = 0

    for curr_amount in range(1, amount + 1):
        for coin in coins:
            if coin <= curr_amount:
                dp_current_minus_coin = dp[curr_amount-coin]
                dp[curr_amount] = min(
                    dp[curr_amount],
                    1 + dp_current_minus_coin
                )

    return dp[amount] if dp[amount] != INF else -1

coins=[2,4]

amount=7

print(coinChange(coins, amount))


def maxProduct(nums: List[int]) -> int:
    max_product = nums[0]
    min_product = nums[0]
    answer = nums[0]

    for num in nums[1:]:
        if num < 0:
            max_product, min_product = min_product, max_product

        max_product = max(num, num*max_product)
        min_product = min(num, num*min_product)
        answer = max(answer, max_product)
    return answer




def wordBreak(s: str, wordDict: List[str]) -> bool:
    n = len(s)

    dp = [False] * (n + 1)

    dp[n] = True

    for i in range(n - 1, -1, -1):

        for word in wordDict:

            if s[i:i+len(word)] == word:

                if dp[i + len(word)]:
                    dp[i] = True
                    break
    return dp[0]

s = "applepenapple"

wordDict = ["apple","pen","ape"]

answer = wordBreak(s, wordDict)

def lengthOfLIS(nums: List[int]) -> int:
    n = len(nums)

    dp = [1] * n
    for i in range(n):
        for j in range(i):

            if nums[j] < nums[i]:
                dp[i] = max(dp[i], 1 + dp[j])

    return max(dp)

nums = [9,1,4,2,3,3,7]
print(lengthOfLIS(nums))

def uniquePaths(m: int, n:int) -> int:
    dp = [1] * n

    for _ in range(1, n):
        for j in range(1, m):
            dp[j] += dp[j - 1]

    return dp[-1]

def longestCommonSubsequence(text1: str, text2: str) -> int:
    answer = 0
    for i in range(len(text1)):
        for j in range(len(text2)):
            if text1[i] == text2[j] and i >= j:
                answer += 1
                break
    return answer

text1 = "bl"
text2 = "yby"
print(longestCommonSubsequence(text1, text2))


def longestCommonSubsequence(text1: str, text2: str) -> int:
    m = len(text1)
    n = len(text2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if text1[i] == text2[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(
                    dp[i][j + 1],
                    dp[i + 1][j]
                )
    return dp[0][0]


def maxSubArray(nums: List[int]):
    current = nums[0]
    answer = nums[0]

    for i in range(1, len(nums)):
        current = max(nums[i], current + nums[i])
        answer = max(answer, current)

    return answer

def canJump(nums: List[int]) -> bool:
    farthest = 0
    for i in range(len(nums)):
        if i > farthest:
            return False
        farthest = max(farthest, i + nums[i])

    return True

def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:

    def helper(big_list):
        big_list.sort(key=lambda x: x[0])
        result_list = []

        for interval in big_list:
            if not result_list:
                result_list.append(interval)
                continue

            if interval[0] <= result_list[-1][1]:
                result_list[-1][1] = max(interval[1], result_list[-1][1])
            else:
                result_list.append(interval)

        return result_list

    intervals.append(newInterval)
    return helper(intervals)


def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
    intervals.sort(key=lambda x:x[1])
    removed = 0
    prev_end = intervals[0][-1]

    for start, end in intervals[1:]:
        if start >= prev_end:
            prev_end = end
        else:
            removed += 1

    return removed


"""
Definition of Interval:
"""

class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end


def canAttendMeetings(intervals: List[Interval]) -> bool:
    if len(intervals) <= 1:
        return True

    intervals.sort(key=lambda x: x.start)

    prev_end = intervals[0].end

    for interval in intervals[1:]:
        if interval.start < prev_end:
            return False
        prev_end = interval.end

    return True

def minMeetingRooms(intervals: List[Interval]) -> int:
    min_heap = []

    if not intervals:
        return 0

    intervals.sort(key=lambda x: x.start)

    for interval in intervals:
        if min_heap and interval.start >= min_heap[0]:
            heapq.heappop(min_heap)

        heapq.heappush(min_heap, interval.end)

    return len(min_heap)


def spiralOrder(matrix: List[List[int]]) -> List[int]:
    if not matrix:
        return []

    result = []

    top = 0
    bottom = len(matrix) - 1
    left = 0
    right = len(matrix[0]) - 1

    while top <= bottom and left <= right:
        for j in range(left, right+1):
            result.append(matrix[top][j])

        top += 1

        for i in range(top, bottom - 1):
            result.append(matrix[i][right])

        right -= 1

        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1

        if left <= right:
            for i in range(bottom, top-1, -1):
                result.append(matrix[i][left])

            left += 1

    return result

def setZeroes(matrix: List[List[int]]) -> None:
    m = len(matrix)
    n = len(matrix[0])

    rows = [False] * m
    cols = [False] * n

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                rows[i] = True
                cols[j] = True

    for i in range(m):
        for j in range(n):
            if rows[i] or cols[j]:
                matrix[i][j] = 0


def hammingWeight(n: int) -> int:
    count = 0
    while n:
        n = n & (n-1)
        count += 1

    return count

def countBits(n: int) -> List[int]:
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)

    return dp

def reverseBits(self, n:int) -> int:
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>=1

    return result


def missingNumber(nums: List[int]) -> int:
    n = len(nums)
    expected = n * (n + 1) // 2
    actual = sum(nums)
    return expected - actual

def missingNumber2(nums: List[int]) -> int:
    n = len(nums)

    for i in range(n):
        n^=i
        n^=nums[i]

    return n

def getSum(a: int, b: int) -> int:
    MASK = 0xFFFFFFFF
    MAX_INT = 0x7FFFFFFF

    while b != 0:
        carry = (a & b) << 1
        a = (a ^ b) & MASK
        b = carry & MASK

    return a if a <= MAX_INT else ~(a ^ MASK)


import heapq
from collections import Counter

def top_k_frequent(stream, k):
    freq = Counter(stream)

    heap = []

    for element, count in freq.items():

        heapq.heappush(heap, (count, element))

        if len(heap) > k:
            heapq.heappop(heap)

    return [element for _, element in heap]


stream = ["A", "B", "A", "C", "B", "A", "D", "C", "A"]
print(top_k_frequent(stream, 2))


def has_cycle(graph):

    state = [0] * len(graph)

    def dfs(node):

        if state[node] == 1:
            return True

        if state[node] == 2:
            return False

        state[node] = 1

        for nei in graph[node]:

            if dfs(nei):
                return True

        state[node] = 2

        return False

    for node in range(len(graph)):

        if dfs(node):
            return True

    return False