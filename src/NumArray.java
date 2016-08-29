/**
 * 72. Range Sum Query - Immutable
 */
public class NumArray {
    int[] sums;
    public NumArray(int[] nums) {
        sums = new int[nums.length];
        int s = 0;
        for(int i = 0; i < nums.length; i++){
            s += nums[i];
            sums[i] = s;
        }
    }

    public int sumRange(int i, int j) {
        if (i == 0) return sums[j];
        return sums[j] - sums[i - 1];
    }
}
