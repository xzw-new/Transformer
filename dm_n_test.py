import torch

# 1. 生成一个随机整数张量
input =torch.randint(1,10,(3,5))
print(f'input: \n{input},shape: {input.shape}')

# 2. 定义掩码张量
mask = torch.tensor([
    [1,2,3,0,0],
    [2,3,4,0,0],
    [2,3,4,1,0]
])

# 3. 处理mask,把非0的值改成1
mask[mask != 0] = 1
print(f'mask: {mask},shape: {mask.shape}')

# 4. 用masked_fill()函数处理input,进行掩码操作
# 参1: 要被处理的变量
# 参2: mask == 0: 布尔条件, 找到mask中值为0的位置
# 参3: 把input中对应mask为0位置的元素,替换成 -1e9(一个非常小的数常用于注意力机制)
result = torch.masked_fill(input,mask ==0,1e-9)

# 5. 打印结果
print(f'result: \n{result},shape: {result.shape}')

