import com.sun.deploy.util.ArrayUtil;

import java.math.BigInteger;
import java.util.*;

/**
 * Created by think on 2016/8/10.
 */
public class Solution {
    public static void main(String[] args){
        TreeNode root = new TreeNode(6);
        root.left = new TreeNode(2);
        root.right = new TreeNode(8);
        root.left.left = new TreeNode(0);
        root.left.right = new TreeNode(4);
        root.left.right.left = new TreeNode(3);
        root.left.right.right = new TreeNode(5);
        root.right.left = new TreeNode(7);
        root.right.right = new TreeNode(9);
        int[] nums={-2, 0, 3, -5, 2, -1};
        System.out.println(new NumArray(nums).sumRange(2,5));

//        System.out.println(lowestCommonAncestor(root,root.left.right.left,null).val);

//        List<List<Integer>> lists = new ArrayList<>(10);
//        lists.add(2,new ArrayList<Integer>());
//        System.out.println(lists.size());
//        lists.get(1);
//        Queue<TreeNode> queue = search(root,root.left.right.left);
//        while(!queue.isEmpty()){
//            System.out.println(queue.poll().val);
//        }
    }
    public static void printArray(int[] nums){
        int l = nums.length;
        for(int i = 0; i < l; i++)
            System.out.print(nums[i]+",");
    }
    /**
     * 1. Reverse String
     */
    public static String reverse(String str){
        //best method
        return new StringBuffer(str).reverse().toString();
    }

    /**
     * 2. Reverse Vowels of a String
     */
    public static String reverseVowels(String s) {
        int left = 0;
        int right = s.length()-1;
        StringBuffer buffer = new StringBuffer(s);
        while(left<right){
            char c_left = buffer.charAt(left);
            while(!isVowel(c_left)){
                left++;
                c_left = buffer.charAt(left);
                if(left>=right) return buffer.toString();
            }
            char c_right = buffer.charAt(right);
            while(!isVowel(c_right)){
                right--;
                c_right = buffer.charAt(right);
                if(right<=left) return buffer.toString();
            }
            buffer.setCharAt(left, c_right);
            buffer.setCharAt(right, c_left);
            left++;
            right--;
        }
        return buffer.toString();
    }
    public static boolean isVowel(char c){
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
                c == 'A'  || c == 'E' || c == 'I' || c == 'O' || c == 'U';
    }


    /**
     * 3. Linked List Random Node. Using reservoir sampling.
     * The probability that result is replaced by ith elements is 1/i in the ith position.
     * In the (i+1)th position, the probability that result is not replaced by
     * the (i+1)th element is i/(i+1),
     * so the probability that the result is ith element is 1/i * i/(i+1) * ... *(n-1)/n = 1/n .
     */
    public static int getRandom(ListNode head) {
        java.util.Random r = new java.util.Random();
        ListNode pos = head;
        int idx = 1;
        int result = -1;
        while(pos != null){
            int randomNum = r.nextInt(idx);
            if(randomNum == idx - 1){
                result = pos.val;
            }
			idx++;
            pos = pos.next;
        }
        return result;
    }

    /**
     * 4. Nim Game
     * When the number of stones is 1-3, you can win, while it is 4, you cannot win.
     * So you should force your friend to face the negative situation that the number is 4.
     * In other words, when the number is 5-7, you can win. When it is 8, you cannot win cuz
     * everyone can remove 1-3 stones in each round. Further, when it is multiple of the number 4,
     * you win.
     */
    public boolean canWinNim(int n) {
        return n % 4 != 0;
    }

    /**
     * 5. Sum of Two Integers
     * Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.
     * First, calculate the number without carry. Second, calculate the carry.
     * e.g. In decimal system, 17+5=12(without carry)+10(carry),
     * in binary system, 101+001=100(without carry)+010(carry).
     * In binary system, XOR can be used to calculate the number without carry.
     * while the carry can be produced only when 1 and 1, so AND coupled with right shift can be
     * used to calculate the carry. If there are more carry, the problem can be solved using recursion.
     */
    public int getSum(int a, int b) {
        if(b == 0) return a;
        int c = (a & b) << 1;
        int d = a ^ b;
        return getSum(d,c);
    }

    /**
     * 6. Add Digits
     * Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
     * The number is called digital root. num=a_k*10^k+a_(k-1)*10^(k-1)+...+a_0=a_k*(9*m_1+1)+...+a_0*(9*0+1).
     * So dr(num)=dr(a_k+...+a_0)=dr(num%9). Note the exception that num is 0.
     */
    public int addDigits(int num) {
        if (num == 0) return 0;
        int r = num % 9;
        return r==0 ? 9 : r;
    }

    /**
     * 7. Maximum Depth of Binary Tree
     */
    public int maxDepth(TreeNode root) {
        return maxDepth(root,0);
    }
    public int maxDepth(TreeNode root, int depth){
        if(root == null) return depth;
        else{
            int leftDepth = maxDepth(root.left,depth+1);
            int rightDepth = maxDepth(root.right,depth+1);
            return (leftDepth > rightDepth) ? leftDepth : rightDepth;
        }
    }

    /**
     * 8. Invert Binary Tree
     *      4
     *    /   \
     *   2     7
     *  / \   / \
     * 1   3 6   9
     * to
     *       4
     *    /   \
     *   7     2
     *  / \   / \
     * 9   6 3   1
     */
    //Recursive
    public TreeNode invertTree(TreeNode root) {
        if(root != null){
            TreeNode node = invertTree(root.right);
            root.right = invertTree(root.left);
            root.left = node;
        }
        return root;
    }
    //Iterative
//    public TreeNode invertTree(TreeNode root) {
//        if (root == null) return null;
//        Queue<TreeNode> queue = new LinkedList<TreeNode>();
//        queue.add(root);
//        while (!queue.isEmpty()) {
//            TreeNode current = queue.poll();
//            TreeNode temp = current.left;
//            current.left = current.right;
//            current.right = temp;
//            if (current.left != null) queue.add(current.left);
//            if (current.right != null) queue.add(current.right);
//        }
//        return root;
//    }

    /**
     * 9. Ransom Note
     * Write  a  function  that  will  return  true  if  the  ransom   note  can  be  constructed  from  the  magazines ;
     * otherwise,  it  will  return  false.
     * Each  letter  in  the  magazine  string  can  only  be  used  once  in  your  ransom  note.
     */
    public static boolean canConstruct(String ransomNote, String magazine) {
        Map<Character,Integer> map = new HashMap<Character, Integer>();
        int len_mgz = magazine.length();
        StringBuffer buf_mgz = new StringBuffer(magazine);
        for(int i = 0; i < len_mgz; i++){
            char letter = buf_mgz.charAt(i);
            if(isLetter(letter) && map.containsKey(letter)) map.put(letter,map.get(letter)+1);
            else map.put(letter,1);
        }
        StringBuffer buf_note = new StringBuffer(ransomNote);
        int len_note = ransomNote.length();
        for(int i = 0; i < len_note; i++){
            char letter = buf_note.charAt(i);
            if(isLetter(letter) && map.containsKey(letter)){
                int currentNum = map.get(letter);
                if(currentNum <= 1){
                    map.remove(letter);
                }else{
                    map.put(letter,currentNum-1);
                }
            }else return false;
        }
        return true;
    }
    public static boolean isLetter(char c){
        return (c >= 'a' && c<='z') || (c >= 'A' && c<='Z');
    }
    //Assume that both strings contain only lowercase letters.
//    public boolean canConstruct(String ransomNote, String magazine) {
//        int[] cnt = new int[26];
//        for (int i = 0; i < magazine.length(); i++) {
//            cnt[magazine.charAt(i) - 'a']++;
//        }
//        for (int i = 0; i < ransomNote.length(); i++) {
//            if(--cnt[ransomNote.charAt(i) - 'a'] < 0) {
//                return false;
//            }
//        }
//        return true;
//    }

    /**
     * 	10. Move Zeroes
     */
    public void moveZeroes(int[] nums) {
        int len = nums.length;
        int zero = 0;
        for(int i = 0; i < len; i++){
            if(nums[i] != 0){
                int tmp = nums[zero];
                nums[zero] = nums[i];
                nums[i] = tmp;
                zero++;
            }
        }
    }

    /**
     * 11. Intersection of Two Arrays
     */
    public static int[] intersection(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int p1 = 0;
        int p2 = 0;
        int len1 = nums1.length;
        int len2 = nums2.length;
        int[] intersection = new int[len1];
        int offset = 0;
        while(p1 < len1 && p2 < len2){
            int n1 = nums1[p1];
            int n2 = nums2[p2];
            if(n1 < n2) p1++;
            else if(n1 > n2) p2++;
            else{
                if(offset ==0) {
                    intersection[offset++] = n1;
                }else{
                    if(intersection[offset-1] < n1) intersection[offset++] = n1;
                }
                p1++;
                p2++;
            }
        }
        return Arrays.copyOf(intersection,offset);
    }

    /**
     * 12. Delete Node in a Linked List
     * Given only access to that node.
     */
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    /**
     * 13. Same Tree
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p != null && q != null){
            if(p.val == q.val){
                return isSameTree(p.left,q.left) && isSameTree(p.right,q.right);
            }else return false;
        }else if(p == null && q == null){
            return true;
        }else return false;
    }

    /**
     * 14. Excel Sheet Column Number
     * Given a column title as appear in an Excel sheet, return its corresponding column number.
     * e.g.
     * A -> 1
     * B -> 2
     * C -> 3
     * ...
     * Z -> 26
     * AA -> 27
     * AB -> 28
     */
    public int titleToNumber(String s) {
        new String(new char[4]).equals(s);
        int len = s.length();
        StringBuffer buffer = new StringBuffer(s);
        int number = 0;
        for(int i = 0; i < len; i++) number = number * 26 + buffer.charAt(i) - 'A' +1;
        return number;
    }

    /**
     * 15. Valid Anagram
     * Given two strings s and t, write a function to determine if t is an anagram of s.
     * e.g.
     * s = "anagram", t = "nagaram", return true.
     * s = "rat", t = "car", return false.
     */
    public boolean isAnagram(String s, String t) {
        int len_s = s.length();
        int len_t = t.length();
        if(len_s != len_t) return false;
        else{
            char[] list_s = s.toCharArray();
            char[] list_t = t.toCharArray();
            Arrays.sort(list_s);
            Arrays.sort(list_t);
            //Following statement can  be replaced by "return Arrays.equals(list_s,list_t)".
            //However following is most efficient.
            return new String(list_s).equals(new String(list_t));
        }
    }
//    public boolean isAnagram(String s, String t) {
//        if (s.length() != t.length()) {
//            return false;
//        }
//        int[] table = new int[26];
//        for (int i = 0; i < s.length(); i++) {
//            table[s.charAt(i) - 'a']++;
//        }
//        for (int i = 0; i < t.length(); i++) {
//            table[t.charAt(i) - 'a']--;
//            if (table[t.charAt(i) - 'a'] < 0) {
//                return false;
//            }
//        }
//        return true;
//    }

    /**
     * 16. Majority Element
     * Given an array of size n, find the majority element.
     * The majority element is the element that appears more than ⌊ n/2 ⌋ times.
     */
    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int range = (n+1) / 2;
        for(int i = 0; i < (n+3)/4; i++){
            if(nums[i] == nums[i + range -1])
                return nums[i];
            if(nums[n-1-i] == nums[n - range + 1])
                return nums[n-1-i];
        }
        return -1;
    }
    //You may assume that the array is non-empty and the majority element always exist in the array.
//    public int majorityElement(int[] nums) {
//        Arrays.sort(nums);
//        return nums[(nums.length +1) / 2];
//    }

    /**
     * 17. Contains Duplicate
     * Given an array of integers, find if the array contains any duplicates.
     * Your function should return true if any value appears at least twice in the array,
     * and it should return false if every element is distinct.
     */
    //Best algorithm
    public boolean containsDuplicate(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i < n - 1; i++){
            if(nums[i] == nums[i + 1]) return true;
        }
        return false;
    }
    //Method using BitSet is better than using HashSet.
    //But the followings are LTE.
//    public static boolean containsDuplicate(int[] nums) {
//        BitSet bt1 = new BitSet();
//        BitSet bt2 = new BitSet();
//        int n = nums.length;
//        for(int i = 0; i < n; i++){
//            int d = nums[i];
//            if(d >= 0){
//                bt1.set(d);
//            }else{
//                bt2.set(-d);
//            }
//        }
//        int c1 = bt1.cardinality();
//        int c2 = bt2.cardinality();
//        if((c1 + c2) == n) return false;
//        else return true;
//    }
//    public boolean containsDuplicate(int[] nums) {
//        Set<Integer> set = new HashSet<Integer>();
//        int n = nums.length;
//        for (int i = 0; i < n; i++){
//            if(!set.add(nums[i])) return true;
//        }
//        return false;
//    }

    /**
     * 18. Given two arrays, write a function to compute their intersection.
     * e.g.
     * Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].
     */
    public int[] intersect(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int len1 = nums1.length;
        int len2 = nums2.length;
        int i1 = 0;
        int i2 = 0;
        int i = 0;
        int[] intersection = new int[len1];
        while(i1 < len1 && i2<len2){
            if(nums1[i1] == nums2[i2]){
                intersection[i++] = nums1[i1];
                i1++;i2++;
            }else if(nums1[i1] > nums2[i2]){
                i2++;
            }else i1++;
        }
        return Arrays.copyOf(intersection,i);
    }

    /**
     * 19. Roman to Integer
     * Given a roman numeral, convert it to an integer.
     */
    public int romanToInt(String s) {
        int len = s.length();
        int result = 0;
        int pre = 0;
        for (int  i = 0; i < len; i++){
            if(i == 0){
                pre = charToInt(s.charAt(i));
                result = pre;
            }else{
                int cur = charToInt(s.charAt(i));
                if (cur <= pre) result += cur;
                else result = result + cur - pre << 1;
                pre = cur;
            }
        }
        return result;
    }
    public int charToInt(char c){
        switch (c){
            case 'I':
                return 1;
            case 'V':
                return 5;
            case 'X':
                return 10;
            case 'L':
                return 50;
            case 'C':
                return 100;
            case 'D':
                return 500;
            case 'M':
                return 1000;
            default:
                return 0;
        }
    }

    /**
     * 20. Reverse Linked List
     */
    public static ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        ListNode nxt = null;
        while(cur != null){
            nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }
    /**
     * 21. Power of Three
     * n is a integer, so n<2^31-1. The max power of 3 is 1162261467.
     */
    public boolean isPowerOfThree(int n) {
        return n > 0 && 1162261467 % n == 0;
    }
    //Using log.
//    public boolean isPowerOfThree(int n) {
//        double epsilon = 10e-15;
//        if(0 == n)
//            return false;
//        double res = Math.log(n)/Math.log(3);
//        //Cause there is error from double.
//        return Math.abs(res - Math.round(res)) < epsilon;
//    }

    /**
     * 22. Power of Two
     * Above solution can be apply to this problem. Following is the other solution.
     */
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
    /**
     * 23. Happy Number
     * A happy number is a number defined by the following process:
     * Starting with any positive integer, replace the number by the sum of the squares of its digits,
     * and repeat the process until the number equals 1 (where it will stay),
     * or it loops endlessly in a cycle which does not include 1.
     * Those numbers for which this process ends in 1 are happy numbers.
     * e.g.
     * 19 is a happy number, 1^2 + 9^2 = 82, 8^2 + 2^2 = 68, 6^2 + 8^2 = 100, 1^2 + 0^2 + 0^2 = 1
     */
    public boolean isHappy(int n) {
        Set<Integer> set = new HashSet<Integer>();
        int s = 0;
        while (n != 1 && set.add(n)) {
            n = squaresOfDigits(n);
        }
        if(n == 1) return true;
        else return false;
    }
    public int squaresOfDigits(int n){
        int sum = 0;
        while(n > 0){
            int digit = n % 10;
            sum += digit * digit;
            n = n/10;
        }
        return sum;
    }

    /**
     * 24. Number of 1 Bits
     * The input is unsigned integer, however there is no unsigned number, so note the negative number.
     */
    public static int hammingWeight(int n) {
        int count = 0;
        if(n < 0){
            count = 1;
            n = n ^ 0x80000000;
        }
        System.out.println(n);
        while(n != 0){
            count += n & 1;
            n = n >> 1;
        }
        return count;
    }

    /**
     * 25. Remove Duplicates from Sorted List
     */
    public ListNode deleteDuplicates(ListNode head) {
        ListNode current = head;
        while (current != null && current.next != null) {
            if (current.next.val == current.val) {
                current.next = current.next.next;
            } else {
                current = current.next;
            }
        }
        return head;
    }
//    public ListNode deleteDuplicates(ListNode head) {
//        if(head == null) return head;
//        ListNode h = head;
//        ListNode c = h;
//        int v = head.val;
//        while(head != null){
//            if(v != head.val){
//                c.next = head;
//                c = c.next;
//                v = head.val;
//            }
//            head = head.next;
//        }
//        c.next = null;
//        return h;
//    }

    /**
     * 26. Lowest Common Ancestor of a Binary Search Tree
     */
    public static TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || p == null || q == null) return null;
        if(root.val > p.val && root.val > q.val) return lowestCommonAncestor(root.left, p, q);
        if(root.val < p.val && root.val < q.val) return lowestCommonAncestor(root.right, p, q);
        return root;
    }
    //Lowest Common Ancestor of a Binary Tree
//    public static TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
//        TreeNode first = p;
//        TreeNode second = q;
//        if(p == null){
//            first = q;
//            second = p;
//        }
//        Queue<TreeNode> queue = search(root,first);
//        while(!queue.isEmpty()){
//            TreeNode node = queue.poll();
//            Queue<TreeNode> list = search(node,second);
//            if(!list.isEmpty()) return node;
//        }
//        return null;
//    }
//    public static Queue<TreeNode> search(TreeNode root, TreeNode p){
//        Queue<TreeNode> queue = new LinkedList<TreeNode>();
//        if(searchNode(root,p,queue)) queue.add(root);
//        return queue;
//    }
//    public static boolean searchNode(TreeNode root, TreeNode p, Queue<TreeNode> queue){
//        if(root == p){
//            return true;
//        }else if (root == null){
//            return false;
//        }
//        else{
//            boolean l = searchNode(root.left,p,queue);
//            boolean r = searchNode(root.right,p,queue);
//            if(l) queue.add(root.left);
//            if(r) queue.add(root.right);
//            return l || r;
//        }
//    }

    /**
     * 27. Climbing Stairs
     * Fibonacci number
     * PROOF:
     * Assume that number of method climbing n stairs is S(n),
     * so climbing n stairs can be treat as climbing n-1 stairs and 1 stair or  climbing n-2 stairs and 2.
     * The former has S(n-1) distinct ways, the latter has S(n-2) distinct ways.
     * S(n) = S(n-1) + S(n-2)
     * This problem can change to climbing 1,2,3... stairs each step. These are all recursion solution.
     */
    public int climbStairs(int n) {
        double constant_a = (1 + Math.sqrt(5)) / 2;
        double constant_b = (1 - Math.sqrt(5)) / 2;
        double constant_c = Math.sqrt(5);
        double value_1 = 0;
        int value_2 = 0;
        if(n > 0)
        {
            value_1 = (Math.pow(constant_a, n+1) - Math.pow(constant_b, n+1)) / constant_c;
            value_2 = (int)value_1;
            return value_2;
        }
        else
        {
            return -1;
        }
    }

    /**
     * 28. Best Time to Buy and Sell Stock
     * Say you have an array for which the ith element is the price of a given stock on day i.
     * If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock),
     * design an algorithm to find the maximum profit.
     */
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int buy = 0;
        int max = 0;
        for(int i = 0; i < n; i++){
            if(i == 0){
                buy = prices[i];
            }else{
                if((prices[i] - buy) > max){
                    max = prices[i] - buy;
                }
                else if((prices[i] - buy) < 0) buy = prices[i];
            }
        }
        return max;
    }

    /**
     * 29. Swap Nodes in Pairs
     * Given a linked list, swap every two adjacent nodes and return its head.
     * */
    public ListNode swapPairs(ListNode head) {
        ListNode currentNode = head;
        ListNode preNode = null;
        while(currentNode != null && currentNode.next != null){
            if(currentNode == head) head = head.next;
            ListNode latter = currentNode.next;
            if(preNode != null) preNode.next = latter;
            currentNode.next = latter.next;
            latter.next = currentNode;
            preNode = currentNode;
            currentNode = currentNode.next;
        }
        return head;
    }
    /**
     * 30. House Robber
     * You are a professional robber planning to rob houses along a street.
     * Each house has a certain amount of money stashed,
     * the only constraint stopping you from robbing each of them is
     * that adjacent houses have security system connected and it will automatically contact the police
     * if two adjacent houses were broken into on the same night.
     * Given a list of non-negative integers representing the amount of money of each house,
     * determine the maximum amount of money you can rob tonight without alerting the police.
     *
     * Dynamic Programming
     * Using recursion.
     * dp[i] refers to the max amount money in ith house, dp[i] = max(dp[i-1], dp[i-2]+nums[i]).
     * */
    public int rob(int[] nums) {
        int odd = 0, even = 0;
        int n = nums.length;
        for(int i = 0; i < n; i++){
            if((i%2)==0){
                even = max(even + nums[i],odd);
            }else{
                odd = max(odd + nums[i], even);
            }
        }
        return max(odd, even);
    }
    public int max(int a,int b){
        return a > b ? a : b;
    }
    /**
     * 31. Power of Four
     * */
    public boolean isPowerOfFour(int num) {
        if((num < 4 && num != 1)|| num == 8 || (num & (num - 1)) != 0) return false;
        int n = (int)Math.sqrt(num);
        return  (n & (n - 1)) == 0;
    }
    /**
     * 32. Ugly Number
     * Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.
     * */
    public boolean isUgly(int num) {
        if(num <= 0) return false;
        num = hasPrime(num, 2);
        num = hasPrime(num, 3);
        num = hasPrime(num, 5);
        return num == 1;
    }
    public int hasPrime(int num, int p){
        while( num % p == 0) num = num / p;
        return num;
    }
    /**
     * 33. Merge Two Sorted Lists
     * */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode head = null;
        ListNode tail = null;
        ListNode p1 = l1;
        ListNode p2 = l2;
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        while (p1 != null && p2 != null){
            if(p1.val < p2.val){
                if(head == null){
                    head = p1;
                    tail = p1;
                }else{
                    tail.next = p1;
                    tail = tail.next;
                }
                p1 = p1.next;
            }else{
                if(head == null){
                    head = p2;
                    tail = p2;
                }else{
                    tail.next = p2;
                    tail = tail.next;
                }
                p2 = p2.next;
            }
        }
        if(p1 != null) tail.next = p1;
        if(p2 != null) tail.next = p2;
        return head;
    }

    /**
     * 34. Linked List Cycle
     * Given a linked list, determine if it has a cycle in it.
     * Let 2 pointers move in different speed, one step and two steps. If there exists cycle,
     * the two pointer will meet in some position.
     * */
    public boolean hasCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
            if(fast == slow) return true;
        }
        return false;
    }

    /**
     * 35. Binary Tree Level Order Traversal II
     * Given a binary tree, return the bottom-up level order traversal of its nodes' values.
     * (ie, from left to right, level by level from leaf to root).
     * */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
       List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int size = queue.size();
            ArrayList<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode head = queue.poll();
                level.add(head.val);
                if (head.left != null) {
                    queue.offer(head.left);
                }
                if (head.right != null) {
                    queue.offer(head.right);
                }
            }
            result.add(level);
        }

        Collections.reverse(result);
        return result;
    }
//    public List<List<Integer>> levelOrderBottom(TreeNode root) {
//        List<List<Integer>> lists = new ArrayList<>();
//        if(root == null) return lists;
//        Queue<RecordNode> queue = new LinkedList<>();
//        Queue<RecordNode> results = new LinkedList<>();
//        int maxLevel = 0;
//        queue.add(new RecordNode(root,0));
//        while (!queue.isEmpty()){
//            RecordNode q = queue.poll();
//            int l = q.level;
//            results.add(q);
//            if(q.node.left != null){
//                queue.add(new RecordNode(q.node.left,l+1));
//                maxLevel = max(l+1,maxLevel);
//            }
//            if(q.node.right != null){
//                queue.add(new RecordNode(q.node.right,l+1));
//                maxLevel = max(l+1,maxLevel);
//            }
//        }
//        for (int i = 0; i <= maxLevel; i++){
//            lists.add(new ArrayList<Integer>());
//        }
//        while(!results.isEmpty()){
//            RecordNode r = results.poll();
//            int l = r.level;
//            List<Integer> list = lists.get(maxLevel - l);
//            list.add(r.node.val);
//        }
//        return lists;
//    }
//    class RecordNode{
//        TreeNode node;
//        int level;
//        RecordNode(TreeNode tn, int l){
//            node = tn;
//            level = l;
//        }
//    }
    /**
     * 36. Remove Element
     * Given an array and a value, remove all instances of that value in place and return the new length.
     * Elements that should not be removed have to be at the front.
     * */
    public int removeElement(int[] nums, int val) {
        int n = nums.length;
        int pos = 0;
        for(int i = 0; i < n; i++){
            if(nums[i] != val){
                nums[pos] = nums[i];
                pos++;
            }
        }
        return pos;
    }

    /**
     * 37. Symmetric Tree
     * Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
     * */
    public boolean isSymmetric(TreeNode root) {
        if(root == null) return true;
        return isEqual(root.left,root.right);
    }
    public boolean isEqual(TreeNode t1, TreeNode t2){
        if(t1 == null && t2 == null) return true;
        else if(t1 != null && t2 != null){
            if(t1.val == t2.val) return isEqual(t1.left,t2.right) && isEqual(t1.right,t2.left);
            else return false;
        }else{
            return false;
        }
    }

    /**
     * 38. Plus One
     * Given a non-negative number represented as an array of digits, plus one to the number.
     * */
    public int[] plusOne(int[] digits) {
        int i = digits.length - 1;
        int carry = 1;
        while (i >= 0 && carry > 0){
            int digit = carry + digits[i];
            if(digit < 10){
                digits[i] = digit;
                carry = 0;
            }else if(digit == 10){
                digits[i] = 0;
                carry = 1;
            }else{
                throw new AssertionError();
            }
            i--;
        }
        if(carry == 1){
            int[] result = new int[digits.length + 1];
            result[0] = 1;
            System.arraycopy(digits,0,result,1,digits.length);
            return result;
        }else{
            return digits;
        }
    }

    /**
     * 39. Balanced Binary Tree
     * Given a binary tree, determine if it is height-balanced.
     * */
    public boolean isBalanced(TreeNode root) {
        int l = level(root);
        if(l < 0) return false;
        return true;
    }
    public int level(TreeNode root){
        if(root == null) return 0;
        int leftLevel = level(root.left);
        int rightLevel = level(root.right);
        if(leftLevel == -1 || rightLevel == -1) return -1;
        int diff = leftLevel - rightLevel;
        if(diff <= 1 && diff >= -1){
            return max(leftLevel,rightLevel)+1;
        }else{
            return -1;
        }
    }

    /**
     * 40. Pascal's Triangle
     */
    public static List<List<Integer>> generate(int numRows) {
        List<List<Integer>> rows = new ArrayList<List<Integer>>();
        for (int i = 0; i < numRows; i++){
            List<Integer> row = new ArrayList<Integer>();
            if(i == 0) row.add(1);
            else{
                List<Integer> pre = rows.get(i-1);
                row.add(1);
                for (int j = 1; j < i; j++){
                    row.add(pre.get(j-1)+pre.get(j));
                }
                row.add(1);
            }
            rows.add(row);
        }
        return rows;
    }
//    public static Integer combination(int n, int k){
//        return factorial(n).divide( factorial(k).multiply(factorial(n-k))).intValue();
//    }
//    public static synchronized BigInteger factorial(int x){
//        ArrayList table = new ArrayList();
//        table.add(BigInteger.valueOf(1));
//        for(int size=table.size();size<=x;size++){
//            BigInteger lastfact= (BigInteger)table.get(size-1);
//            BigInteger nextfact= lastfact.multiply(BigInteger.valueOf(size));
//            table.add(nextfact);
//        }
//        return (BigInteger) table.get(x);
//    }

    /**
     * 41. Binary Tree Level Order Traversal
     * */
    public List<List<Integer>> levelOrder(TreeNode root) {
        Stack<Integer> stack = new Stack<Integer>();
        stack.push(1);
        stack.peek();
        List<List<Integer>> lists = new ArrayList<List<Integer>>();
        if(root == null) return lists;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            List<Integer> list = new ArrayList<Integer>();
            for(int i = 0; i < size; i++){
                TreeNode node = queue.poll();
                list.add(node.val);
                if(node.left != null) queue.add(node.left);
                if(node.right != null) queue.add(node.right);
            }
            lists.add(list);
        }
        return lists;
    }

    /**
     * 43. Remove Duplicates from Sorted Array
     * */
    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        if(n == 0) return 0;
        int offset = 0;
        for(int i = 1; i < n; i++){
            if(nums[i] != nums[offset]){
                offset++;
                nums[offset] = nums[i];
            }
        }
        return offset+1;
    }

    /**
     *  44. Factorial Trailing Zeroes
     * */
    public int trailingZeroes(int n) {
        int i = n/5;
        int r = i;
        while(i >= 5){
            i = i/5;
            r+=i;
        }
        return r;
    }

    /**
     * 45. Pascal's Triangle II
     * Given an index k, return the kth row of the Pascal's triangle.
     * */
    public List<Integer> getRow(int rowIndex) {
        List<Integer> pre = new ArrayList<Integer>();
        List<Integer> row = new ArrayList<Integer>();
        for (int i = 0; i <= rowIndex; i++){
            if(i == 0) pre.add(1);
            else{
                row.add(1);
                for (int j = 1; j < i; j++){
                    row.add(pre.get(j-1)+pre.get(j));
                }
                row.add(1);
                pre = row;
                row = new ArrayList<Integer>();
            }
        }
        return pre;
    }

    /**
     * 46. Palindrome Number
     * Determine whether an integer is a palindrome. Do this without extra space.
     * */
    public boolean isPalindrome(int x) {
        char[][] b = new char[9][9];
        if (x < 0) return false;
        int div = 1;
        while ( x / div >= 10) div = div * 10;
        while (x != 0){
            int high = x / div;
            int low = x % 10;
            if (high != low) return false;
            x = (x % div) / 10;
            div = div /100;
        }
        return true;
    }

    /**
     * 47. Valid Sudoku
     * 9 X 9
     * Determine if a Sudoku is valid.
     * */
    public boolean isValidSudoku(char[][] board) {
        for (int i = 0; i < 9; i++){
            char[] row = new char[]{1,2,3,4,5,6,7,8,9};
            char[] col = new char[]{1,2,3,4,5,6,7,8,9};
            char[] block = new char[]{1,2,3,4,5,6,7,8,9};
            for (int j = 0; j < 9; j++){
                char c1 = board[i][j];
                char c2 = board[j][i];
                boolean f = isValid(c1, row) && isValid(c2, col);
                if(!f) return false;
            }
            int j = i / 3;
            int k = i % 3;
            int cs = k * 3;
            int ce = (k + 1) * 3;
            int rs = j * 3;
            int re = (j + 1) * 3;
            for (int r = rs; r < re; r++){
                for (int c = cs; c < ce; c++){
                    char ch = board[r][c];
                    if(!isValid(ch,block)) return false;
                }
            }
        }
        return true;
    }
    public boolean isValid(char n, char[] nums){
        if( n == '.') return true;
        int i = n - '0' -1;
        if(nums[i] == 0) return false;
        nums[i] = 0;
        return true;
    }

    /**
     * 48. Path Sum
     * Given a binary tree and a sum, determine if the tree has a root-to-leaf path
     * such that adding up all the values along the path equals the given sum.
     * */
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root == null) return false;
        int rest = sum - root.val;
        if(root.left == null && root.right == null){
            if(rest == 0) return true;
            else return false;
        }
        return hasPathSum(root.left, rest) || hasPathSum(root.right, rest);
    }

    /**
     * 49. Bulls and Cows
     * You are playing the following Bulls and Cows game with your friend:
     * You write down a number and ask your friend to guess what the number is.
     * Each time your friend makes a guess, you provide a hint that indicates how many
     * digits in said guess match your secret number exactly in both digit and position (called "bulls")
     * and how many digits match the secret number but locate in the wrong position (called "cows").
     * Your friend will use successive guesses and hints to eventually derive the secret number.
     * */
    public String getHint(String secret, String guess) {
        int size = secret.length();
        char[] digits_s = secret.toCharArray();
        char[] digits_g = guess.toCharArray();
        int bulls = 0;
        int cows = 0;
        for (int i = 0; i < size; i++){
            if(digits_g[i] == digits_s[i]) bulls++;
        }
        Arrays.sort(digits_g);
        Arrays.sort(digits_s);
        int ps = 0;
        int pg = 0;
        while (ps < size && pg < size){
            if (digits_g[pg] == digits_s[ps]){
                cows++;
                pg++;
                ps++;
            }else if (digits_g[pg] > digits_s[ps]){
                ps++;
            }else{
                pg++;
            }
        }
        cows = cows - bulls;
        String r = bulls+"A"+cows+"B";
        return r;
    }

    /**
     * 50. Minimum Depth of Binary Tree
     * */
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null){
            return 1;
        }
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        if (left != 0 && right != 0){
            return min(left,right)+1;
        }else{
            return left+right+1;
        }
    }
    public int min(int a, int b){
        return a < b ? a : b;
    }

    /**
     * 51. Guess Number Higher or Lower
     * We are playing the Guess Game. The game is as follows:
     * I pick a number from 1 to n. You have to guess which number I picked.
     * Every time you guess wrong, I'll tell you whether the number is higher or lower.
     * You call a pre-defined API guess(int num) which returns 3 possible results (-1, 1, or 0):
     * */
    /* The guess API is defined in the parent class GuessGame.
   @param num, your guess
   @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
      int guess(int num); */
    public int guessNumber(int n) {
        int start = 1;
        int end = n;
        int g = avg(start,end);
        int i = guess(g);
        while (i != 0 && start < end){
            if (i < 0){
                end = g;
            }else{
                start = g;
            }
            g = avg(start,end);
            i = guess(g);
        }
        return g;
    }
    //the method is provided by leetcode
    public int guess(int g){
        return 0;
    }
    public static int avg(int a, int b){
        return a + (b - a + 1)/2;
    }

    /**
     * 52. Binary Tree Paths
     * */
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> result = new ArrayList<>();
        if(root == null) return result;
        if (root.left == null && root.right == null){
            result.add(""+root.val);
            return result;
        }
        List<String> lefts = binaryTreePaths(root.left);
        List<String> rights = binaryTreePaths(root.right);
        for (int i = 0; i < lefts.size(); i++){
            result.add(root.val+"->"+lefts.get(i));
        }
        for (int i = 0; i < rights.size(); i++){
            result.add(root.val+"->"+rights.get(i));
        }
        return result;
    }
    /**
     * 53. Isomorphic Strings
     * Notice that the types of character is fixed. Remember pre-position of each character.
     * */
    public boolean isIsomorphic(String s, String t) {
        int[] cnt_s = new int[256];
        int[] cnt_t = new int[256];
        int size = s.length();
        for(int i = 0; i < size; i++){
            int chs = s.charAt(i);
            int cht = t.charAt(i);
            if(cnt_s[chs] != cnt_t[cht]) return false;
            cnt_s[chs] = i + 1;
            cnt_t[cht] = i + 1;
        }
        return true;
    }

    /**
     * 54. Rectangle Area
     * */
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int s1 = (C - A) * (D - B);
        int s2 = (G - E) * (H - F);
        int a3 = 0;
        int b3 = 0;
        if(A < G && G <= C){
            if(A <= E) a3 = G - E;
            else a3 = G - A;
        }else if(A <= E && E < C){
            a3 = C - E;
        }else if(E <= A && C <= G){
            a3 = C - A;
        }
        if(a3==0) return s2 + s1;
        else{
            if(B < H && H <= D){
                if(B <= F) b3 = H - F;
                else b3 = H - B;
            }else if(B <= F && F < D){
                b3 = D - F;
            }else if(F <= B && D <= H){
                b3 = D - B;
            }
            return s1 + s2 - a3 * b3;
        }
    }
    /**
     * 56. Remove Nth Node From End of List
     * Given a linked list, remove the nth node from the end of list and return its head.
     * */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode start = head;
        ListNode p = head;
        while(n > 0){
            p=p.next;
            n--;
        }
        if(p == null) return head.next;
        while(p.next != null){
            start = start.next;
            p = p.next;
        }
        start.next = start.next.next;
        return head;
    }

    /**
    * 57. Word Pattern
     * Given a pattern and a string str, find if str follows the same pattern.
    * */

    public boolean wordPattern(String pattern, String str) {
        HashMap<Character,String> mappings = new HashMap<>();

        String[] words = str.split(" ");
        char[] patternLetters = pattern.toCharArray();

        if(words.length != patternLetters.length)
            return false;

        for(int i = 0; i < patternLetters.length; i++)
        {
            if(mappings.get(patternLetters[i]) != null && !mappings.get(patternLetters[i]).equals(words[i]))
                return false;
            else if(mappings.get(patternLetters[i]) == null && mappings.containsValue(words[i]))
                return false;
            else
                mappings.put(patternLetters[i], words[i]);
        }

        return true;
    }
//    public boolean wordPattern(String pattern, String str) {
//        int[] p = new int[26];
//        int[] wdp = new int[26];
//        String[] words = str.split("\\s+");
//        char last = str.charAt(str.length()-1);
//        int add = 0;
//        if(words[0].isEmpty()) add++;
//        if (pattern.length() != words.length-add) return false;
//        else {
//            if(last == ' ') return false;
//            LetterSet set = new LetterSet();
//            for (int i = 0; i < pattern.length(); i++){
//                int x = pattern.charAt(i) - 'a';
//                int y = set.get(words[i + add]);
//                if(y==-1) return false;
//                if(p[x] != wdp[y]) return false;
//                p[x] = i + 1;
//                wdp[y] = i + 1;
//            }
//        }
//        return true;
//    }
//    class LetterSet{
//        String[] set = new String[26];
//        int cur = 0;
//        public int get(String s){
//            int k = search(s);
//            if(k>=0) return k;
//            else{
//                if(cur >= 26) return -1;
//                set[cur++] = s;
//                return cur-1;
//            }
//        }
//        int search(String s){
//            for(int i = 0; i <cur; i++){
//                if(set[i].equals(s)) return i;
//            }
//            return -1;
//        }
//    }

    /**
     * 58. Merge Sorted Array
     * Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
     * You may assume that nums1 has enough space (size that is greater or equal to m + n) to
     * hold additional elements from nums2.
     * The number of elements initialized in nums1 and nums2 are m and n respectively.
     * */
    public static void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1, j = n - 1, index = m + n -1;
        while(i >= 0 && j >= 0){
            if (nums1[i] > nums2[j]){
                nums1[index--] = nums1[i--];
            }else{
                nums1[index--] = nums2[j--];
            }
        }
        while(i >= 0){
            nums1[index--] = nums1[i--];
        }
        while(j >= 0){
            nums1[index--] = nums2[j--];
        }
    }

    /**
     * 59. 	Intersection of Two Linked Lists
     * */
    public static ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode a = headA;
        ListNode b = headB;
        while (a != b){
            a = (a == null ? headB : a.next);
            b = (b == null ? headA : b.next);
        }
        return a;
    }

    /**
     * 60. Contains Duplicate II
     * Given an array of integers and an integer k, find out whether there are two distinct indices i and j
     * in the array such that nums[i] = nums[j] and the difference between i and j is at most k.
     * */
    public static boolean containsNearbyDuplicate(int[] nums, int k) {
        int n = nums.length;
        if (n < 1) return false;
        int min = nums[0];
        int max = nums[0];
        for (int i = 1; i < n; i++){
            if(nums[i] < min) min = nums[i];
            if(nums[i] > max) max = nums[i];
        }
        int[] map = new int[max-min+1];
        Arrays.fill(map,-1);
        for (int i = 0; i< n; i++){
            int x = nums[i] - min;
            if(map[x] == -1){
                map[x] = i;
                continue;
            }else{
                if (i - map[x] <= k) return true;
                else map[x] = i;
            }
        }
        return false;
    }
//    public static boolean containsNearbyDuplicate(int[] nums, int k) {
//        Map<Integer,Integer> map = new HashMap<>();
//        int n = nums.length;
//        for(int i = 0; i < n; i++){
//            if(map.containsKey(nums[i]) && i - map.get(nums[i]) <= k)
//                return true;
//            else{
//                map.put(nums[i],i);
//            }
//        }
//        return false;
//    }

    //The method is more efficient than above int test, but LTE as submission.
//    public static boolean containsNearbyDuplicate(int[] nums, int k) {
//        Set<Integer> set = new HashSet<>();
//        int n = nums.length;
//        for(int i = 0; i < n; i++){
//            if(i > k){
//                set.remove(nums[i-k-1]);
//            }
//            if(!set.add(nums[i])){
//                return true;
//            }
//        }
//        return false;
//    }

    /**
     * 61. First Unique Character in a String
     * */
    public int firstUniqChar(String s) {
        int[] map = new int[26];
        int n = s.length();
        if(n == 0) return -1;
        Arrays.fill(map,-n);
        for (int i = 0; i < n; i++){
            int c = s.charAt(i) - 'a';
            if(map[c] == -n){
                map[c] = i;
            }else if(map[c] >= 0){
                map[c] = -i;
            }else{
                map[c] = -i;
            }
        }
        int min = n;
        for(int i = 0; i < 26; i++){
            if(map[i] >= 0 && map[i] < min){
                min = map[i];
            }
        }
        if(min == n) return -1;
        return min;
    }

    /**
     * 62. Count and Say
     * The count-and-say sequence is the sequence of integers beginning as follows:
     * 1, 11, 21, 1211, 111221, ...
     * 1 is read off as "one 1" or 11.
     * 11 is read off as "two 1s" or 21.
     * 21 is read off as "one 2, then one 1" or 1211.
     * Given an integer n, generate the nth sequence.
     * */
    public String countAndSay(int n) {
        int i = 0;
        String oldString = "1";
        while (i < n){
            StringBuilder builder = new StringBuilder();
            char pre='0';
            int count = 1;
            for (int j = 0; j < oldString.length(); j++){
                char c = oldString.charAt(j);
                if(j == 0) pre = c;
                else{
                    if(c != pre){
                        builder.append(""+count+pre);
                        count = 1;
                        pre = c;
                    }else{
                        count++;
                    }
                }
            }
            builder.append(""+count+pre);
            oldString = builder.toString();
            i++;
        }
        return oldString;
    }

    /**
     * 63. Valid Parentheses
     * */
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++){
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c== '{'){
                stack.push(c);
            }else{
                if(stack.isEmpty()) return false;
                char t = stack.pop();
                if ((c == '}' && t != '{')
                        || (c == ']' && t != '[')
                        || (c == ')' && t != '(')){
                    return false;
                }
            }
        }
        if(stack.isEmpty()) return true;
        else return false;
    }

    /**
     * 64. Length of Last Word
     * Given a string s consists of upper/lower-case alphabets and empty space characters ' ',
     * return the length of last word in the string.
     * If the last word does not exist, return 0.
     * */
    public int lengthOfLastWord(String s) {
        int n = s.length()-1;
        int count = 0;
        boolean start = false;
        while(n >= 0){
            if (s.charAt(n) != ' '){
                count++;
                start = true;
            }else{
                if(start) break;
            }
            n--;
        }
        return count;
    }

    /**
     * 65. Palindrome Linked List
     * */
    public boolean isPalindrome(ListNode head) {
        if(head == null || head.next == null) return true;
        ListNode slow = head;
        ListNode fast = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode l2 = null;
        if(fast == null){
            l2 = slow;
        }else{
            l2 = slow.next;
        }
        ListNode l1 = reverse(head,slow);
        while(l1 != null && l2 != null){
            if(l1.val != l2.val) return false;
            l1 = l1.next;
            l2 = l2.next;
        }
        if(l1 == null && l2 == null) return true;
        return false;
    }
    public ListNode reverse(ListNode head, ListNode end){
        ListNode pre = null;
        ListNode cur = head;
        ListNode nxt = null;
        while(cur != end){
            nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }

    /**
     * 66. Remove Linked List Elements
     * Remove all elements from a linked list of integers that have value val.
     * */
    public ListNode removeElements(ListNode head, int val) {
        ListNode p = head;
        ListNode newHead = head;
        ListNode pre = null;
        while (p != null){
            if (pre == null){
                if(p.val == val){
                    newHead = p.next;
                }
                else{
                    pre = p;
                }
            }else{
                if(p.val == val){
                    pre.next = p.next;
                }else{
                    pre = p;
                }
            }
            p = p.next;
        }
        return newHead;
    }

    /**
     * 67. Reverse Bits
     * */
    public static int reverseBits(int n) {
        int result = 0;
        int num = n;
        for (int i = 0; i < 32; i++){
            int bit = num & 0x00000001;
            System.out.println(bit);
            if (i == 0) result = bit;
            else{
                result = (result << 1) + bit;
                System.out.println("r:"+result);
            }
            num = num >>> 1;
        }
        return result;
    }

    /**
     * 68. Longest Common Prefix
     * */
    public String longestCommonPrefix(String[] strs) {
        int n = strs.length;
        if( n == 0 ) return "";
        else if(n == 1) return strs[0];
        Arrays.sort(strs);
        String first = strs[0];
        String last = strs[n-1];
        for(int i=0; i<first.length(); i++) {
            if(first.charAt(i) != last.charAt(i))
                return first.substring(0,i);
        }
        return first;
    }
//    public String longestCommonPrefix(String[] strs) {
//        StringBuilder builder = new StringBuilder();
//        int pos = 0;
//        int len = min(strs);
//        while(pos < len){
//            char c = '\0';
//            int i = 0;
//            while (i < strs.length){
//                if (i == 0) c = strs[i].charAt(pos);
//                else{
//                    if (strs[i].charAt(pos) != c) break;
//                }
//                i++;
//            }
//            if (i == strs.length){
//                builder.append(c);
//                pos++;
//            }
//            else return builder.toString();
//        }
//        return builder.toString();
//    }
//    public int min(String[] strs){
//        if (strs.length == 0) return 0;
//        int min = Integer.MAX_VALUE;
//        for(int i = 0; i < strs.length; i++){
//            if(min > strs[i].length()) min = strs[i].length();
//        }
//        return min;
//    }

    /**
     * 69. Add Binary
     * */
    public static String addBinary(String a, String b) {
        String result = "";
        char carry = '0';
        int n1 = a.length() - 1;
        int n2 = b.length() - 1;
        StringBuilder builder = new StringBuilder();
        while(n1 >=0 && n2 >= 0){
            int count = 0;
            char d = '0';
            char d1 = a.charAt(n1--);
            char d2 = b.charAt(n2--);
            if(d1 == '1') count++;
            if(d2 == '1') count++;
            if(carry == '1') count++;
            if(count >= 2){
                carry = '1';
                if(count == 3) d = '1';
                else d = '0';
            }else{
                carry = '0';
                if(count == 1) d = '1';
                else d = '0';
            }
            builder.append(d);
        }
        if(carry == '1'){
            while (n1 >= 0){
                if(a.charAt(n1--) == '1') builder.append('0');
                else{
                    carry = '0';
                    builder.append('1');
                    result = a.substring(0,n1+1) + builder.reverse().toString();
                    break;
                }
            }
            while (n2 >= 0){
                if(b.charAt(n2--) == '1') builder.append('0');
                else{
                    carry = '0';
                    builder.append('1');
                    result = b.substring(0,n2+1) + builder.reverse().toString();
                    break;
                }
            }
            if(carry == '1') result = builder.append('1').reverse().toString();
        }else{
            if(n1 >= 0) result = a.substring(0,n1+1) + builder.reverse().toString();
            else if(n2 >= 0)  result = b.substring(0,n2+1) + builder.reverse().toString();
            else result = builder.reverse().toString();
        }
        return result;
    }

    /**
     * 70. Implement strStr()
     * Returns the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
     * */
    public int strStr(String haystack, String needle) {
        int i = 0;
        int n = haystack.length();
        int l = needle.length();
        if (n < l) return -1;
        while(n - i >= l){
            String sub = haystack.substring(i,n);
            if(sub.startsWith(needle)) return i;
            i++;
        }
        return -1;
    }

    /**
     * 71. Two Sum
     * Given an array of integers, return indices of the two numbers such that they add up to a specific target.
     * You may assume that each input would have exactly one solution.
     * */
    public int[] twoSum(int[] nums, int target) {
        int n = nums.length;
        if(n < 2) return new int[]{-1,-1};
        else{
            Map<Integer,Integer> map = new HashMap<>();
            for(int i = 0; i < n; i++){
                if(map.isEmpty()){
                    map.put(target - nums[i],i);
                }else{
                    int num = nums[i];
                    if(map.containsKey(num)){
                        return new int[]{map.get(num),i};
                    }else{
                        map.put(target - nums[i],i);
                    }
                }
            }
        }
        return new int[]{-1,-1};
    }
}
