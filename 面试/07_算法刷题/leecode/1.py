
def maxScore(cardPoints, k: int) -> int:
    cnt = left = 0
    n = len(cardPoints)
    l = n - k
    value = sum(cardPoints[:])
    if l <= 0:
        return value 

    for i, x in enumerate(cardPoints):
        cnt += x

        left = i - l + 1
        if left < 0:
            continue
        
        value = min(value, cnt)

        cnt -= cardPoints[left]

    return sum(cardPoints) - value


print(maxScore([9,7,7,9,7,7,9], 7))

