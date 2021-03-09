import numpy as np

class Solution(object):
    def __init__(self):
        self.res = []
        self.dig2char = {"2":["a","b","c"], "3":["d","e","f"], "4":["g","h","i"],
                        "5":["j","k","l"],"6":["m","n","o"], "7":["p","q","r","s"],
                        "8":["t","u","v"],"9":["w","x","y","z"]}

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if len(digits)==0:
            return []
        track = []
        self.combinate(digits, track)
        return self.res

    def combinate(self,digits, track):
        #结束条件
        if len(track) == len(digits):
            self.res.append("".join(track))
            return
        for c in self.dig2char.get(digits[len(track)]):
            # if c in track:
            #     continue
            track.append(c)
            self.combinate(digits, track)
            track.pop()

s = Solution()
print(s.letterCombinations("23"))
