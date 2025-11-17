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
        seen.add(text[start])
        max_len = max(max_len, end-start + 1)
    return max_len

#####################################################################################################################################################################################################################
from typing import List
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
print(threeSum(nums))


#####################################################################################################################################################################################################################