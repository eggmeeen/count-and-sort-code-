#include <stdio.h>
void merge_and_count(int arr[], int left, int mid, int right, int temp[]) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }

}

void merge_sort_and_count(int arr[], int left, int right, int temp[]) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        merge_sort_and_count(arr, left, mid, temp);
        merge_sort_and_count(arr, mid + 1, right, temp);

        merge_and_count(arr, left, mid, right, temp);
    }
}
int main() {
    int k;
    scanf("%d", &k);//number of test cases
    int n;
    scanf("%d", &n);
    int arr[n]; 
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }
    int temp[n];
    merge_sort_and_count(arr, 0, n - 1, temp);
    for (int i = n - k; i < n; i++) {
        printf("%d", arr[i]);
        if (i != n - 1) {
            printf(" ");
        }
    }
    printf("\n");
}