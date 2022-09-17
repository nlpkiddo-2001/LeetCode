package leetcode;

public class MedainOfTwoSorted {
    public static void main(String[] args) {
            double[] nums1 = {1,2};
            double[] nums2 = {3,4};
        System.out.println(medianOfTwoSorted(nums1,nums2));
    }
    public static double medianOfTwoSorted(double[] nums1, double[] nums2){
        double n = nums1.length;
        double m = nums2.length;
        int i = 0, j = 0, k = 0;
        double [] result = new double[nums1.length+ nums2.length];
        double start = 0;
        double end = result.length-1;
        double mid = start + end / 2;
        while(i<n&&j<m){
            if(nums1[i]<nums2[j])
                result[k++] = nums1[i++];
            else
                result[k++]=nums2[j++];
        }
        while(i< nums1.length)
            result[k++]=nums1[i++];
        while(j< nums2.length)
            result[k++]=nums2[j++];
        double resultMid = 0.0f;
        if(result.length%2!=0)
            resultMid = result[(int) mid];
        else if(result.length%2==0)
            resultMid = (result[(int) mid] + result[(int) (mid+1)] )/ 2;


        return resultMid;
        }

    }

