package org.example;

import java.util.*;

class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

public class LeetCoding {
    public static void main(String[] args) {
        int numBottles = 10;
        int numExchange = 3;
        System.out.println(maxBottlesDrunk(numBottles, numExchange));

    }

    public static ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode res = new ListNode(0, head);
        ListNode dummy = res;

        for (int i = 0; i < n; i++) {
            head = head.next;
        }

        while (head != null) {
            head = head.next;
            dummy = dummy.next;
        }

        dummy.next = dummy.next.next;

        return res.next;
    }

    public static int maxBottlesDrunk(int numBottles, int numExchange) {
        int totalBottles = numBottles;
        while (numBottles >= numExchange) {
            numExchange += 1;
            totalBottles += numBottles / numExchange;
            int numDivideByExchange = numBottles / numExchange;
            int numModuloByExchange = numBottles % numExchange;
            numBottles = numDivideByExchange + numModuloByExchange;

        }
        return totalBottles;
    }

    public static int numWaterBottles(int numBottles, int numExchange) {
        /**
         15 / 4 => 3.7
         so 15 bottle + 3 set of empty bottle + reminder 3
         from 3 set of empty bottle we get 3 full bottle
         now empty that again, totally 6. now 6 - 4 => 2 empty bottle + 1 full bottle
         now empty that full bottle , totaly 3 empty bottle=> which cannot be filled
         so 15 + 3 + 1 = 19.
         */

        int totalBottles = numBottles;
        while (numBottles >= numExchange) {
            totalBottles += numBottles / numExchange;
            int numDivideByExchange = numBottles / numExchange;
            int numModuloByExchange = numBottles % numExchange;
            numBottles = numDivideByExchange + numModuloByExchange;

        }
        return totalBottles;
    }

    public static int chalkReplacer(int[] chalk, int k) {
        /**
         while k should not equals 0
         i pointer = 0
         check if i == chalk.len
         if so i = 0
         chalk[i] - k;
         */
        long totalChalk = 0;
        for (int c : chalk) {
            totalChalk += c;
        }

        // Reduce k by the total chalk used in complete rounds
        k = (int) (k % totalChalk);

        // Find the student who will need to replace the chalk
        for (int i = 0; i < chalk.length; ++i) {
            if (chalk[i] > k) {
                return i;
            }
            k -= chalk[i];
        }

        // This point should never be reached due to constraints
        return 0;
    }

    public static ListNode mergeNodes(ListNode head) {
        ListNode dummy = new ListNode(0);
        ListNode newListTail = dummy;

        ListNode current = head.next;
        int sumVal = 0;

        while (current != null) {
            if (current.val == 0) {
                newListTail.next = new ListNode(sumVal);
                newListTail = newListTail.next;
                sumVal = 0;
            } else {
                sumVal += current.val;
            }

            current = current.next;
        }

        return dummy.next;

    }

    public static List<Integer> findSubstring(String s, String[] words) {
        ArrayList<String> all_combinations = (ArrayList<String>) helper(words);
        Set<Integer> uniqueIndices = new HashSet<>();
        for (String combination : all_combinations) {
            int index = s.indexOf(combination);
            while (index >= 0) {
                uniqueIndices.add(index);
                index = s.indexOf(combination, index + 1);
            }
        }
        List<Integer> result = new ArrayList<>(uniqueIndices);
        return result;
    }

    public static List<String> helper(String[] strs) {
        List<String> result = new ArrayList<>();
        boolean[] used = new boolean[strs.length];
        generatePermutations(strs, used, new StringBuilder(), result);
        return result;
    }

    private static void generatePermutations(String[] strs, boolean[] used, StringBuilder current, List<String> result) {
        if (current.length() == getTotalLength(strs)) {
            result.add(current.toString());
            return;
        }

        for (int i = 0; i < strs.length; i++) {
            if (!used[i]) {
                used[i] = true;
                current.append(strs[i]);
                generatePermutations(strs, used, current, result);
                current.setLength(current.length() - strs[i].length());
                used[i] = false;
            }
        }
    }

    private static int getTotalLength(String[] strs) {
        int length = 0;
        for (String str : strs) {
            length += str.length();
        }
        return length;
    }

    public static int minimizeMax(int[] nums, int p) {
        List<int[]> differences = new ArrayList<>();

        for (int i = 0; i < nums.length - 1; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                int difference = Math.abs(nums[i] - nums[j]);
                differences.add(new int[]{i, j, difference});
            }
        }

        differences.sort(Comparator.comparingInt(a -> a[2]));

        boolean[] used = new boolean[nums.length];
        List<int[]> selectedPairs = new ArrayList<>();

        for (int[] diff : differences) {
            if (selectedPairs.size() == p) break;
            int i = diff[0];
            int j = diff[1];

            if (!used[i] && !used[j]) {
                selectedPairs.add(diff);
                used[i] = true;
                used[j] = true;
            }
        }

        return selectedPairs.isEmpty() ? 0 : selectedPairs.get(selectedPairs.size() - 1)[2];
    }

    public static int minDifference(int[] nums) {
        if (nums.length <= 4) {
            return 0;
        }
        Arrays.sort(nums);
        int ans = nums[nums.length - 1] - nums[0];
        for (int i = 0; i <= 3; i++) {
            int curr_idx = nums.length - (3 - i) - 1;
            ans = Math.min(ans, nums[curr_idx] - nums[i]);
        }
        return ans;

    }

    public static int[] intersect(int[] nums1, int[] nums2) {
        int l1 = nums1.length;
        int l2 = nums2.length;
        int i = 0, j = 0, k = 0;
        while (i < l1 && j < l2) {
            if (nums1[i] < nums2[j]) {
                i++;
            } else if (nums2[j] < nums1[i]) {
                j++;
            } else {
                nums1[k++] = nums1[i++];
                j++;
            }
        }
        return Arrays.copyOfRange(nums1, 0, k);
    }

    public static ArrayList<Integer> bfsOfGraph(int V, ArrayList<ArrayList<Integer>> adj) {
        // Your implementation of bfsOfGraph here (assuming it's already provided)

        ArrayList<Integer> result = new ArrayList<>();
        boolean[] visited = new boolean[V];
        Queue<Integer> q = new LinkedList<Integer>();

        q.add(0);
        visited[0] = true;

        while (!q.isEmpty()) {
            Integer node = q.poll();
            result.add(node);

            for (Integer it : adj.get(node)) {
                if (!visited[it]) {
                    visited[it] = true;
                    q.add(it);
                }
            }
        }
        return result;
    }

    public static ArrayList<Integer> BFS(int V, ArrayList<ArrayList<Integer>> adjacent) {
        ArrayList<Integer> result = new ArrayList<>();
        boolean[] visited = new boolean[V];
        Queue<Integer> queue = new LinkedList<>();

        queue.add(0);
        visited[0] = true;

        while (!queue.isEmpty()) {
            Integer node = queue.poll();
            result.add(node);

            for (Integer it : adjacent.get(node)) {
                if (!visited[it]) {
                    visited[it] = true;
                    queue.add(it);
                }
            }
        }
        return result;
    }

    public static String multiply(String num1, String num2) {
        int len1 = num1.length();
        int len2 = num2.length();
        int[] result = new int[len1 + len2];

        // Iterate through num1 and num2 from right to left
        for (int i = len1 - 1; i >= 0; i--) {
            for (int j = len2 - 1; j >= 0; j--) {
                int digit1 = num1.charAt(i) - '0';
                int digit2 = num2.charAt(j) - '0';
                int product = digit1 * digit2;

                // Add the product to the correct position in result array
                int sum = product + result[i + j + 1];
                result[i + j + 1] = sum % 10;  // current digit
                result[i + j] += sum / 10;     // carry
            }
        }

        // Convert result array to string
        StringBuilder sb = new StringBuilder();
        for (int digit : result) {
            if (!(sb.length() == 0 && digit == 0)) { // skip leading zeros
                sb.append(digit);
            }
        }

        // Handle case where result is "0"
        if (sb.length() == 0) {
            return "0";
        }

        return sb.toString();
    }

    public static int[] getNoZeroIntegers(int n) {
        int[] result = new int[2];
        result[0] = 1;
        result[1] = n - result[0];
        if (helper(result[0]) || helper(result[1])) {
            while (helper(result[0]) || helper(result[1])) {
                result[0] = result[0] + 1;
                result[1] = n - result[0];
            }
        }

        return result;
    }

    public static boolean helper(int n) {
        String numStr = String.valueOf(n);
        return numStr.contains("0");
    }

    public static void sortColors(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        int current = 0;

        while (current <= right) {
            if (nums[current] == 0) {
                swap(nums, current, left);
                left++;
                current++;
            } else if (nums[current] == 2) {
                swap(nums, current, right);
                right--;
            } else {
                current++;
            }

        }
    }

    private static void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public static boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] dp_array = new boolean[m + 1][n + 1];

        dp_array[0][0] = true;

        for (int i = 1; i <= n; i++) {
            if (p.charAt(i - 1) == '*') {
                dp_array[0][i] = dp_array[0][i - 1];
            }
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp_array[i][j] = dp_array[i][j - 1] || dp_array[i - 1][j];
                } else if (p.charAt(j - 1) == '?' || s.charAt(i - 1) == p.charAt(j - 1)) {
                    dp_array[i][j] = dp_array[i - 1][j - 1];
                }
            }
        }

        return dp_array[m][n];
    }

    public static long countSubarrays(int[] nums, int k) {
        long count = 0;
        Map<Integer, Integer> frequencyMap = new HashMap<>();
        int start = 0, maxFreq = 0;
        int currentMax = Integer.MIN_VALUE;

        for (int end = 0; end < nums.length; end++) {
            // Update the frequency of the current element
            frequencyMap.put(nums[end], frequencyMap.getOrDefault(nums[end], 0) + 1);
            // Update the current maximum and its frequency
            if (nums[end] > currentMax || frequencyMap.get(nums[end]) > maxFreq) {
                currentMax = nums[end];
                maxFreq = frequencyMap.get(nums[end]);
            }

            // Once the frequency of the current maximum reaches k, count the subarrays
            while (maxFreq >= k) {
                count += nums.length - end; // Add all possible extensions of this subarray
                // Reduce the frequency of the starting element and possibly adjust currentMax and maxFreq
                frequencyMap.put(nums[start], frequencyMap.get(nums[start]) - 1);
                if (nums[start] == currentMax) {
                    maxFreq--;
                    if (maxFreq < k) {
                        currentMax = Integer.MIN_VALUE;
                        for (int i = start + 1; i <= end; i++) {
                            if (frequencyMap.get(nums[i]) > maxFreq) {
                                maxFreq = frequencyMap.get(nums[i]);
                                currentMax = nums[i];
                            } else if (nums[i] == currentMax) {
                                maxFreq = frequencyMap.get(nums[i]); // Update maxFreq if we encounter the currentMax again
                            }
                        }
                    }
                }
                start++; // Contract the window from the left
            }
        }
        return count;
    }

    public static int climbStairs(int n) {
        if (n <= 1) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        return dp[n];
    }

    public static boolean canJump(int[] nums) {
        int furthest = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > furthest) {
                return false;
            }
            furthest = Math.max(furthest, i + nums[i]);
            if (furthest >= nums.length - 1) {
                return true;
            }
        }
        return false;
    }

    public static int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int j = 0; j < n; j++) {
            dp[0][j] = 1;
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }

        return dp[m - 1][n - 1];
    }

    public static int minOperations(int[] nums, int k) {
        int n = nums.length;
        int ans = 0;
        for (int i = 0; i <= 30; i++) {
            int b = 0;
            for (int j = 0; j < n; j++) {
                if (((nums[j] >> i) & 1) == 1) {
                    b = b ^ 1;
                }
            }
            if (((k >> i) & 1) != b) ans++;
        }
        return ans;
    }

    public static void print_pattern(int n) {
        for (int i = n; i >= 1; i--) {
            int space = n - i;
            for (int j = 1; j <= space; j++) {
                System.out.print(" ");
            }
            for (int k = 1; k < i; k++) {
                System.out.print("* ");
            }
            System.out.println();
        }
    }

    public List<Integer> findSubstringOptimal(String s, String[] words) {
        List<Integer> result = new ArrayList<>();
        if (s == null || s.length() == 0 || words == null || words.length == 0) {
            return result;
        }

        int wordLength = words[0].length();
        int wordsCount = words.length;
        int totalLength = wordLength * wordsCount;

        Map<String, Integer> wordFrequency = new HashMap<>();
        for (String word : words) {
            wordFrequency.put(word, wordFrequency.getOrDefault(word, 0) + 1);
        }

        for (int i = 0; i <= s.length() - totalLength; i++) {
            Map<String, Integer> seenWords = new HashMap<>();
            int j = 0;
            while (j < wordsCount) {
                int wordIndex = i + j * wordLength;
                String word = s.substring(wordIndex, wordIndex + wordLength);
                if (wordFrequency.containsKey(word)) {
                    seenWords.put(word, seenWords.getOrDefault(word, 0) + 1);
                    if (seenWords.get(word) > wordFrequency.get(word)) {
                        break;
                    }
                } else {
                    break;
                }
                j++;
            }
            if (j == wordsCount) {
                result.add(i);
            }
        }

        return result;
    }

    public int numberOfPairs(int[] nums1, int[] nums2, int k) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int count = 0;
        for (int i = 0; i < len1; i++) {
            for (int j = 0; j < len2; j++) {
                if (nums2[j] * k != 0 && nums1[i] % (nums2[j] * k) == 0) {
                    count += 1;
                }
            }
        }
        return count;
    }
}