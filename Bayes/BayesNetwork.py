from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
student_model = BayesianModel([('D', 'G'), ('I', 'G'), ('I', 'S'), ('G', 'L')]) #构建网络结构
gradeCPT = TabularCPD(    #构建成绩节点的条件概率表
    variable='G', #节点名称
    variable_card=3, #节点取值个数
    values=[[0.3, 0.05, 0.9, 0.5], #条件概率表
            [0.4, 0.25, 0.08, 0.3],
            [0.3, 0.7, 0.02, 0.2]],
    evidence=['I', 'D'], #该节点的依赖节点
    evidence_card= [2, 2] #依赖节点的取值个数
)

difficultyCPT = TabularCPD(  #构建考试难度节点的条件概率表
    variable='D', # 节点名称
    variable_card=2, #该节点的取值个数
    values=[[0.6],
            [0.4]] #节点值
)

intelCPT=TabularCPD(  #构建智力节点条件概率表
    variable='I', #节点名称
    variable_card=2,#节点取值个数
    values=[[0.7],
            [0.3]]#节点取值
)

SATCPT=TabularCPD( #构建SAT节点条件概率表
    variable='S', # 节点名称
    variable_card=2, #节点取值个数
    values=[[0.95, 0.2],
            [0.05, 0.8]],#节点条件概率表
    evidence=['I'],#依赖节点
    evidence_card=[2]#依赖节点的取值个数
)
letterCPT = TabularCPD( #构建letter节点的条件概率表
    variable='L',  #节点名称
    variable_card=2, #节点取值个数
    values=[[0.1, 0.4, 0.99],
            [0.9, 0.6, 0.01]], #条件概率表
    evidence=['G'], #依赖节点
    evidence_card=[3] #依赖节点的取值个数
)

student_model.add_cpds(gradeCPT,
                       difficultyCPT,
                       intelCPT,
                       SATCPT,
                       letterCPT)

print(student_model.get_cpds())
print(student_model.get_independencies())
student_infer = VariableElimination(student_model)
p_grade = student_infer.query(
    variables=['L'],
    evidence={'D':1, 'I':0, 'G':0}
)

print(p_grade)