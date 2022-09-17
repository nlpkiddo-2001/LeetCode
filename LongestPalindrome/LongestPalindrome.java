package leetcode;

public class LongestPlaindrome {
    public static void main(String[] args) {
        String s = "malayalam";//racecar
        System.out.println(longestPalindrome(s));
    }
    public  static String longestPalindrome(String s) {
        if(s==null&&s.length()<1)
            return "";
        int start = 0;
        int end = 0;
        for(int i = 0; i < s.length();i++){
            int con1 = expanderString(s,i,i);
            int con2 = expanderString(s,i,i+1);
            int finalLen = Math.max(con1,con2);
            if(finalLen>end - start){
                start = i - ((finalLen-1) / 2);
                end = i + (finalLen / 2);
            }
        }
        return s.substring(start, end+1);

    }
    public static int expanderString (String s, int left , int right){
        if(s==null||left>right)
            return 0;
        while(left>=0 &&right<s.length() &&s.charAt(left)==s.charAt(right)){
            left--;
            right++;
        }
        return right - left - 1;
    }

}
