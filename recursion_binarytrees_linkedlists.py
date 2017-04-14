import math

def factorial(number):
    if number == 1:
        return 1
    else:
        return number * factorial(number-1)
        
print(factorial(5))


def palindrome(word):
    if len(word) <= 1:
        return True
    else:
        return ((word[0] == word[-1]) and palindrome(word[1:-1]))
    
    
def merge(s1, s2):
    '''merges 2 strings aplhabetically'''
    if s1 == '' or s2 == '' :
        return s1 + s2
    if s1[0] <= s2[0]:
        return s1[0] + merge(s1[1:], s2)
    else:
        return s2[0] + merge(s1, s2[1:])


def same_string(word1, word2):
    if ( word1 == '' and word2 == ''):
        return True
    elif (word1[0] != word2[0]):
        return False
    else:
        return same_string(word1[1:], word2[1:])
    
    
def almost_same_string(word1, word2):
    if ( word1 == '' and word2 == ''):
        return True
    elif (word1[0] != word2[0]):
        return same_string(word1[1:], word2[1:])
    else:
        return almost_same_string(word1[1:], word2[1:])
    
    
    
def sumr(L):
    if len(L) == 1:
        return L[0]
    else:
        return L[0] + sumr(L[1:])
        
        

def average(L):
    if len(L) < 1:
        return L[0]
    else:
        return sumr(L) / len(L)
    
    
    
def sum_nested(L):
    '''Return the sum of all the numbers in the nested list L.'''
    if L == []:
        return float('inf')
    if isinstance(L[0], int):
        return L[0] + sum_nested(L[1:])
    else:
        return sum_nested[L[0]] + L[1:]



def count_normal(L, number):
    while len(L) == 0:
        for elements in L:
            if elements > number:
                result = elements
        return result

    
    
def count(L, number):
    if len(L) == 0:
        return 0
    else:
        if (L[0] >= number):
            return  1 + count(L[1:], number)
        else:
            return count(L[1:], number)
        

def palind(word):
    reverse = word[::-1]
    if word == reverse:
        return True
    else:
        return False
    
    
def near_palind2(s):
    if not s:
        return True
    elif s[0] == s[-1]:
        return near_palind2(s[1:-1])
    else:
        return palindrome(s[1:-1])
    
def reverse(L):
    if L == [] :
        return []
    else:
        return (L[-1]) + reverse(L[:-1])
    
def same_length(s1, s2):
    if s1 == '' or s2 == '':
        return s1 == '' and s2 == ''
    else:
        return same_length(s1[1:] , s2[1:])
    
def reverse(n):
    if len(n) == 1:
        return n
    else:
        return n[-1] + reverse(n[:-1])
    
    
def add_comma(n):
    if n <= 3:
        return n
    else:
        return str((add_comma(n[:-3]))+ ',' + n[-3:])
    

    

def length(L):
    if L == []:
        return 0
    else:
        return 1 + length(L[1:])

def sum_even(n):
    if n < 1:
        return 0
    elif n%2 != 0:
        return sum_even(n-1)
    else:
        return n + sum_even(n-2)
    
    
    
def sum_total(L):
    new = []
    for elements in L:
        if element% 2 == 0:
            result += new.append(element)
        return result
    
    
    
def prod(L):
    if L == []:
        return 1
    elif len(L) == 1:
        return L[0]
    else:
        return L[0] * prod(L[1:])
    
def sum1to_n(n):
    if n==1:
        return 1
    else:
        n + sum1to_n(n-1)
        
def rmax(L):
    if len(L) == 1:
        return L[0]
    if L[0] > rmax(L[1:]):
        return L[0]
    else:
        return rmax(L[1:])
    
def rmin(L, n):
    if n == 1:
        return L[0]
    elif L[0] < rmin(L[1:], n-1):
        return L[0]
    else:
        return rmin(L[1:], n-1)
    

    
    

    '''
def min_bst(root):
    
    curr = root
    parent = None
    if root < 1:
        return 1
    else:
        curr = parent
        curr = curr.left
        if root[0] > min_bst([1:]):
            return root[0]
        else:
            return min_bst([1:])
    '''
        
def smallest(L):
    if len(L) == 1:
        return L[0]
    else:
        if L[0] < smallest(L[1:]):
            return L[0]
        else:
            return smallest(L[1:])
        
def second_smallest(L):
    if len(L) == 1:
        return L
    else:
        if L[0] < smallest(L[1:]):
            if L[0] < second_smallest(L[1:]):
                return second_smallest(L[1:])
            
        else:
            return smallest(L[1:])
        
        
def weird_version(s):
    """ (str) -> str
    Return a weird version of string s.
    """
    if len(s) < 2:
        return s
    else:
        return s[1] + weird_version(s[2:]) + s[0]
if __name__ == "__main__":
    print(weird_version("A48WEIRD"))
    print(weird_version("ABC12345DEF"))
    

##there is an error with the duplicate function##
def duplicate(s):
    if len(s) < 2:
        return False
    elif s[0] == s[-1]:
        return True
    else:
        return duplicate(s[1:]) == duplicate(s[:-1])
    
 
def divide(s, chr1):
    if s == '' :
        return s
    if s[0] < chr1:
        return s[0] + divide(s[1:], chr1)
    else:
        return divide(s[1:], chr1) + s[0]
    
def length(L):
    if len(L) < 1:
        return 0
    else:
        return 1 + length(L[1:])
    
class UnequalLists(Exception):
    '''
    '''

def dotproduct(L1, L2):
    if len(L1) and len(L2) < 2:
        return L1[0] * L2[0]
    elif len(L1) == 0 and len(L2) == 0:
        return []
    elif len(L1) != len(L2):
        raise UnequalLists('unequal lists')
    else:
        return dotproduct(L1[1:], L2[1:]) + (L1[0] * L2[0])
    
    
def addtwolists(l1, l2):
    list1 = []
    if (len(l1) and len(l2)) < 2:
        result =  l1[0] + l2[0]
        
    elif len(l1) == 0 and len(l2) == 0:
        result = []
    elif len(l1) != len(l2):
        raise UnequalLists('unequal lists')
    else:
        result = l1[0] + l2[0], addtwolists(l1[1:], l2[1:])
    return result


def edit_distance(s1, s2):
    if len(s1) == 0:
        result = 0
    else:
        result = edit_distance(s1[1:], s2[1:])
        if s1[0] != s2[0]:
            result += 1
    return result


def subsequence(s1, s2):
    ''' return true iff s2 can be made equal to s1 by removing 0 or more of its character'''
    if (len(s1) != 0 and (len(s2) == 0)):
        return False
    elif len(s1) == 0 or (s2 == s1):
        return True
    else:
        if s1[-1] == s2[-1]:
            return subsequence(s1[:-1], s2[:-2])
        else:
            return subsequence(s1, s2[:-1])



def rmax_nested(s):
    '''(list of int) -> int
    return the maximum number in a given list
    REQ:
    len(s) >= 1
    '''
    if s == []:
        return float('-inf')
    else:
        if isinstance(s[0], int):
            largest = s[0]
        else:
            largest = rmax(s[0])
        if largest > rmax(s[1:]):
            result = largest
        else:
            result = rmax(s[1:])
        return result
    
def sum_nested(L):
    if len(L) == 0:
        return 0
    else:
        if isinstance(L[0], list):
            return sum_nested(L[0]) + sum_nested(L[1:])
        else:
            return sum_nested(L[1:]) + L[0]
        
def linear_search(L, x):
    if L:
        if L[0] == x:
            return True
        else:
            return linear_search(L[1:], x)
    else:
        return False

def binary_search(L, s):
    ''' Return True if s in L using recursion'''
    if len(L) == 0:
        return False
    else:
        mid = len(L) // 2
        mid_element = L[mid]
        if s == mid_element:
            return True
        elif(mid_element < s):
            return binary_search(L[mid + 1:], s)
        else:
            return binary_search(L[:mid], s)
        
def binary_search2(L,s):
    ''' return the index of s in the list L, or -1 if s is not in L'''
    if len(L) == 0:
        result = -1
    else:
        mid_index = len(L)//2
        mid_element = L[mid_index]
        if mid_element == s:
            result = L[mid_index]
        elif(mid_element > s):
            result = binary_search2(L[0:mid_index], s)
        else:
            result = binary_search2(L[mid_index+1:], s)
        if (result != -1):
            result = result + mid_index + 1
    return result


###############################################
## TREES ###################

def tree_average(root):
    value = tree_avg_helper(root)
    total = value[0]
    counter = value[1]
    return total/counter




def tree_avg_helper(root, total, counter):
    if (not root.left and not root.right):
        total = root.value
        counter = 1
    else:
        total += root.value
        counter += 1
    if(root.left):
        left = tree_avg_helper(root.left)
        total += left[0]
        counter += left[1]
    if(root.right):
        right = tree_avg_helper(root.right)
        total += right[0]
        counter += right[1]
    return total, counter

def depth_tree(root):
    if root is None:
        return 0
    else:
        return 1 + max(depth_tree(root.left), depth_tree(root.right))
                       
def sum_tree(root):
    if root is None:
        return None
    else:
        return root.data + sum_tree(root.left) + sum_tree(root.right)
    
def is_top_heavy(root):
    if root:
        
        if root.left or root.right == None:
            return True
        elif root.left.data > root.data and root.right.data > root.data:
            return True and is_top_heavy(root.right) and is_top_heavy(root.left)
        else:
            False
    else:
        False
        
        
        
        
    #    
def swap_nodes(root, k):
    '''Given a binary tree and integer value k, the task is to swap sibling
    nodes of every k’th level where k >= 1'''
    swap_nodes_helper(root, 1, k)
        

def swap_nodes_helper(root, level, k):
    '''Given a binary tree and integer value k, the task is to swap sibling
    nodes of every k’th level where k >= 1'''
    if (root is None or (root.left is None and root.right is None)):
        return
    if (level + 1)%k == 0:
        root.left, root.right = root.right, root.left
        
    swap_nodes_helper(root.left, level+1, k)
    swap_nodes_helper(root.right, level+1, k)
  #  



    ### out of nowhere
def reverse_stack(s):
    '''Given a Stack, s, recursively print the contents of
    the stack in reverse'''
    if (s.isempty()):
        return
    else:
        ele = s.pop()
        reverse_stack(s)
        print(ele)
        return
    ###
    
###  BST  ###
    

def is_bst(t):
    ''' Checks whether the tree is BST or not'''
    if root:
        if t.data >= t.left.data and t.data<= t.right.data:
            return True and is_bst(t.left) and is_bst(t.right)
        else:
            return False
    else:
        return False    


def insert(self, value):
    if value < self.data :
        self.left = self.left.insert(value)
    else:
        self.right = self.right.insert(value)
    return self

def search(self, value):
    '''return true if the value is in the tree'''
    if value == self.data:
        return True
    elif value<self.data:
        return self.left.search(value)
    else:
        return self.right.search(value)
    
    
def delete(self, value):
    if self.data == value:
        if self.left.isempty():
            self = self.right
        elif self.right.isempty():
            self = self.left
        else:
            self.data = self.right._find_smallest()
            self.right = self.right.delete(self.data)
    elif value < self.data:
        self.left = self.left.delete(value)
    else:
        self.right = self.right.delete(value)
    return self


def _find_smallest(self):
    if self.left.isempty():
        return self.data
    else:
        return self.left._find_smallest()
    
    
#### BINARY TREE ####
def visit(self):
    print(self.data)
    

def pre_order(root):
    if root != None:
        root.visit()
        pre_order(root.left)
        pre_order(root.right)
        
def post_order(root):
    if root != None:
        post_order(root.left)
        post_order(root.right)
        root.visit()
        
def in_order(root):
    in_order(root.left)
    root.visit()
    in_order(root.right)
    
def search(root, value):
    '''return true if value is in the tree'''
    if root == None:
        return False
    else:
        return (root.data == value) or search(root.left, value) or search(root.right, value)
    
def insert(root, value):
    '''return the root of the tree with value inserted'''
    if root == None:
        return BinTreeNode(value)
    else:
        if root.left == None:
            root.left = BinTreeNode(value)
        elif root.right == None:
            root.right = BinTreeNode(value)
        else:
            root.right = insert(root.right, value)
        return root

def delete(root, value):
    '''return the root of the tree with value deleted'''
    if root == None:
        return None
    if root.data == value:
        if root.left == None:
            root = root.right
        elif root.right == None:
            root = root.left
        else:
            root.data = _find_leaf(root.right)
            root.right = delete(root.right, root.data)
    else:
        root.right = delete(root.right, value)
        root.left = delete(root.left, value)
    return root

def _find_leaf(root):
    ''' return the data value of the leaf node in the tree'''
    if root.left == None and root.right == None:
        return root.data
    elif root.right != None:
        return _find_leaf(root.right)
    else:
        return _find_leaf(root.left)


def num_at_depth(root, k):
    '''find the number of element in the k th depth'''
    if root is None:
        return 1
    if k == '0':
        return 1
    if k == '1':
        return 2
    else:
        total = 0
        total += num_at_depth(root.left, k-1) + num_at_depth(root.right, k-1)
        return total

def sum_depths(r):
    return sum_depths_help(r, 0)

def sum_depths_help(r, depth):
    if(r is None):
        result = 0
    elif(r.left is None and r.right is None):
        result = depth
    else:
        left = sum_depths_help(r.left, depth + 1)
        right = sum_depths_help(r.right, depth + 1)
        result = left + right + depth
    return result




##
def areidentical(root1, root2):
    if root1 is None and root2 is None:
        return True
    if root1 is None or root2 is None:
        return False
    return (root1.data == root2.data and areidentical(root1.left, root2.left) and
            areidentical(root1.right, root2.right))

def is_subtree(root, s):
    if s is None:
        return True
    if root is None:
        return True
    if areidentical(root, s):
        return True
    return is_subtree(root.left, s) or is_subtree(root.right, s)

def second_largest(root):
    if root:
        if root.left != None:
            left_max = root.data + max(second_largest(root.left))
        elif root.right != None:
            right_max = root.data + max(second_largest(root.right))
        if left_max > right_max:
            return right_max
        else:
            return left_max
    else:
        return 0

#---------------------------------------------------------------------


### LINKED LIST ####
def linked_list_reverse (item, tail):
    next = item.next
    item.next = tail
    if next is None:
        return item
    else:
        return linked_list_reverse(next, item)    
    
    
    
    
    
    
def add_tail(self, new_obj, current):
    if (self.head == None):
        self.add_front(new_obj)
    elif (current  == None):
        current = self.head
    if (current.next == None):
        new_node = Node(new_obj, None)
        current.next = new_node
    else:
        self.add_tail(new_obj, current.next)
        
        
    
def add_front(self, new_obj):
    ''' insert the new_obj at the head of the list)'''
    new_node = Node(new_obj, self.head)
    self.head = new_node
    
        
def add_after(self, new_obj, after_obj):
    '''add new obj after the first occurence of the after obj)'''
    LinkedList.add_after_helper(self.head, new_obj, after_obj)
    

def add_after_helper(current, new_obj, after_obj):
    '''insert new_obj after the after_obj in the list of nodes starting with current'''
    if current != None:
        if current.data == after_obj:
            new_node = Node(new_obj, current.next)
            current.next = new_node
        else:
            LinkedList.add_after_helper(current.next, new_obj, after_obj)
    
def remove_tail(self):
    ''' remove last element in the list and return it.'''
    if self.head.next == None:
        return_val = self.head
        self.head = None
        return return_val
    else:
        return LinkedList.remove_tail_helper
    
def remove_tail_helper(current):
    '''remove and return the linked lists tail node starting from current'''
    if current.next.next == None:
        return_val = current.next
        current.next = None
        return val

def find(self, obj):
    if self.head != None:
        return LinkedList.find_helper(self.head, obj)
    
def find_helper(current, obj):
    '''return the node containing object obj'''
    if current.data == obj:
        return obj
    else:
        return LinkedList.find_helper(current.next, obj)
    
    
def find_different(head, o):
    ''' return a node with data o in list with first node head.'''
    if head == None or head.data == o:
        result = head
    else:
        result = find_different(head.next, o)
    return result

def find_opt_param(self, obj, current):
    '''return the node in the linked list containing object where current is
    current node we are looking at.'''
    if current == None:
        current = self.head
    if current != None:
        if current.data == obj:
            return current
        elif current.next != None:
            return self.find_opt_param(obj, current.next)


def find_previous(self, obj):
    '''return the node whose link points to the first node with data
    equal to obj after the current node.'''
    if self.head != None:
        return LinkedList.find_previous_helper(self.head, obj)



def find_previous_helper(current, obj):
    '''return the node whose link points to the first node with data
    equal to obj after the current node.'''
    if current.next == None:
        return None
    elif current.next.data == obj:
        return current
    else:
        return LinkedList.find_previous_helper(current.next, obj)
    

def remove(self, obj):
    '''remove the node with data obj'''
    if self.head == None:
        return 
    elif self.head.data == obj:
        self.head = self.head.next
    else:
        current = self.find_previous(obj)
        if current != None:
            current.next = current.next.next
            
def delete(head, o):
    '''remove a node with data o from list'''
    (at_node, before_node) = find2(head, o)
    if head == None:
        result = None
    elif head.data == o:
        result = head.next
    else:
        head.next = delete(head.next, p)
        result = head
    return result

def move2front(head, o):
    '''move a node with data o list with first node head to the front'''
    (at_node, before_node) = find2(head,o)
    if at_node != None and before_node != None:
        before_node.next = at_node.next
        at_node.next = head
        head = at_not
    return head
            
            
def find2(head, o):
    '''return 2 nodes, one with data o in list with first node head,
    and the second is the node, before it.'''
    if head == None:
        (at_node, before_node) = (None, None)
    elif head.data == o:
        (at_node, before_node) = (head, None)
    else:
        (at_node, before_node) = find2(head.next, o)
        if at_node == head.next != None:
            before_node = head
    
def contains(head, o):
    '''return True if the object is in the linked list'''
    if head.data == None:
        return False
    else:
        result = (head.data == o) or contains(head.next, o)
    return result



# can be improved using merge sort.
def linked_list_sort(head):
    if head == None:
        result = head
    elif head.data < head.next.data:
        return head
    else:
        curr = head
        head = head.next
        head.next = curr
        return linked_list_sort(head.next)
   
   
##  **** good one***
    
def linked_list_merge(L1, L2):
    ''' Merge two sorted linked lists'''
    if L1 == None:
        return L2
    if L2 == None:
        return L1
    if L1.data < L2.data:
        L1.next = linked_list_merge(L1.next, L2)
        return L1
    else:
        L2.next = linked_list_merge(L1, L2.next)
        return L2
        
        
def list_split(L):
    ''' split sorted linked list'''
    if L == None:
        L1 = L2 = None
        return (L1, L2)
    
    if L.order == '1': # assuming the order is given
        L1 = L
        (L1.next, L2) = list_split(L.next)
        
    else:
        L2 = L
        (L1, L2.next) = list_split(L.next)
    return (L1, L2)

def merge_lists(L1, L2):
    if len(L1) == 0:
        return L2
    if len(L2) == 0:
        return L1
    if len(L1) and len(L2) == 0:
        return []
    if L1[0] <= L2[0]:
        return [L1[0]] + merge_lists(L1[1:], L2)
    else:
        return [L2[0]] + merge_lists(L1, L2[1:])
    