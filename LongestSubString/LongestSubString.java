package leetcode;

public class LongestSubString {
    public static void main(String[] args) {
            String str = "abcabcbb";
        System.out.println(longest(str));
    
    }
    public static int longest(String str){
        String test = "";
        int maxLength = -1;
        if(str.isEmpty())
            return 0;
        if(str.length()==1)
            return 1;
        for(char c  : str.toCharArray()){
            String current = str.valueOf(c);
            if(test.contains(current)){
                test = test.substring(test.indexOf(current)+1);
            }
            test = test + String.valueOf(c);
            maxLength = Math.max(test.length(),maxLength);
        }
        return maxLength;
    }
}
