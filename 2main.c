#include <stdio.h>
#include <stdlib.h>
void find_max_min(int array[], int size,int *max, int *min) {
     *max = array[0];
     *min = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i]> *max) {
            *max = array[i];
        }
        if (array[i] < *min) {
            *min = array[i];
        }
    }
}
void sort_array(int array[], int size, int max, int min) {
    int *temp = (int *)calloc(max - min + 1, sizeof(int)); // allocating memory for temp array
    for (int i = 0; i < size; i++) {
        temp[array[i] - min ]++ ;
    }

    for (int i = 0, j = 0; i <= max - min; i++) {
        while (temp[i] > 0) {
            array[j] = i + min;
            j++;
            temp[i]--;
        }
    }
    free(temp); // freeing memory
}
void no_num_biggerthan_k(int array[], int size,int k) {
    if (k > size) {
        k = size;
    }
    for (int i = size - k;i < size; i++)
    {
        printf("%d ", array[i]);
    }
}
int main() {
    int k;
    scanf("%d", &k);
    int n;
    scanf("%d", &n);
    int array[n];
    for (int i = 0; i < n; i++) {
        scanf("%d", &array[i]);
    }
    int max, min;
    find_max_min(array, n, &max, &min);
    sort_array(array, n, max, min);
    no_num_biggerthan_k(array, n, k);
    printf("\n");
    return 0;
}