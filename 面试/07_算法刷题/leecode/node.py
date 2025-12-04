class ListNode:
    """定义链表节点"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList_recursion(head: ListNode) -> ListNode:
    if head is None or head.next is None:
        return head

    new_head = reverseList_recursion(head.next)

    head.next.next = head
    head.next = None

    return new_head



if __name__ == "__main__":
    # 示例：创建链表 1->2->3->None
# 使用列表存储节点值，通过循环构建链表
    values = [1, 2, 3,4,5]
    dummy = ListNode(0)  # 创建虚拟头节点，简化边界处理
    current = dummy
    for val in values:
        current.next = ListNode(val)  # 创建新节点并连接
        current = current.next  # 移动指针到新节点
    head = dummy.next  # 真正的头节点是虚拟节点的下一个

    # 调用递归函数反转链表
    new_head = reverseList_recursion(head)

    # 打印反转后的链表：3->2->1->None
    current = new_head
    while current:
        print(current.val, end=" -> " if current.next else " -> None\n")
        current = current.next

        