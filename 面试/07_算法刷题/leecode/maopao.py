def bubble_sort_basic(arr):
    n = len(arr)
    for i in range(n):
        swapper= False
        for j in range(0, n - i - 1):
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapper = True

        if not swapper:
            break
    return arr

# 测试
my_list = [64, 34, 25, 12, 22, 11, 90]
sorted_list = bubble_sort_basic(my_list)
print("排序后的数组:", sorted_list)  # 输出: [11, 12, 22, 25, 34, 64, 90]