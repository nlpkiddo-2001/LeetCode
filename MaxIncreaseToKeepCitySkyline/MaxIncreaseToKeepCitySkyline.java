class Solution {
    public int maxIncreaseKeepingSkyline(int[][] grid) {
        
        int len = grid.length;
        int[] row = new int[len];
        int[] col = new int[len];
        for(int i = 0; i<len;i++){
            for(int j=0;j<len;j++){
                row[i]=Math.max(grid[i][j],row[i]);
                col[j]=Math.max(grid[i][j],col[j]);
            }
        }
        int totalSum = 0;
        for(int i = 0; i < len;i++){
            for(int j = 0; j < len;j++){
                 int tempSum = Math.min(row[i],col[j])-grid[i][j];
                totalSum+=tempSum;
            }
        }
    return totalSum;
}
}
