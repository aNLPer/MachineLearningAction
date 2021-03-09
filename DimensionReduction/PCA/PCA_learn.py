"""
首先，我们先从生活的场景出发感受一下降维的身影，然后我们会从二维数据的降维例子看看降维在做什么事情，
从宏观的层面介绍PCA在完成什么样的任务，
然后我会从数学的微观层面上讲解PCA是如何完成降维任务的，
最后我们会用纯Python实现PCA算法完成鸢尾花数据集的分类，
接着会调用sklearn的pca工具来做一个人脸识别的降维分析，看看PCA到底在实战任务中是怎样的一个存在。
https://mp.weixin.qq.com/s/uAlBtGTmtBSjcnp9bWQr5Q
"""
# res = []
# def permute(nums):
#     tarck = []
#     backtrack(nums, tarck)
#     return res
#
#
# def backtrack(nums, track):
#     #结束条件
#     if (len(track) == len(nums)):
#         res.append(track.copy())
#         print('当前res是: ' + str(res))
#         return
#
#     for i in nums:
#         #做选择
#         if i in track:
#             continue
#         track.append(i)
#
#         backtrack(nums, track)
#         track.pop()
#
#
# res = permute([1, 2, 3])
# print(res)
res = []

def permute(nums):
    track = []
    trackback(nums, track)
    #return res

def trackback(nums, track):
    #结束条件
    if len(track) == len(nums):
        res.append(track.copy())
        return

    for i in nums:
        # 选择路径
        if i in track:
            continue
        track.append(i)
        trackback(nums, track)
        #撤销选择
        track.pop()

# nums = [1, 2, 3]
# permute(nums)
# print(res)
a = ["1", "2", "3"]
print("".join(a))
