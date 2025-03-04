import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
 
def calc_probability(e, price, cost, profit, index_i=[0], index_j=[0], index_m=[0], max_prob=[0], index_n=[0], min_prob=[0]): 
    """
    e: 浮点型，最低概率阈值，要小于1
    price: 列表，各个档位的价格分布
    cost: 浮点型，用户单次抽奖花费
    profit: 浮点型，从用户单次抽奖中抽取的利润期望
    index_i: 列表，位置列表，[i1,i2,...,in]
    index_j: 列表，位置列表，[j1,j2,...,jn],第i1个位置的概率大于等于对应的第j1个位置的概率
    """
    z = (-1) * np.array(price)  #转化为最小值解法
    
    #一些异常判断
    if len(index_i) != len(index_j):
        return "第一类个性化设置中，位置对应列表长度要一致"
    
    for l in range(len(index_i)) :
        if index_i[l] not in range(0,len(price)+1) or index_j[l] not in range(0,len(price)+1):
            return '第一类个性化设置中，位置选取超出价位列表长度'
        
    if len(index_m) != len(max_prob):
        return "第二类个性化设置中，对应列表长度要一致"
    
    for l in range(len(index_m)) :
        if index_m[l] not in range(0,len(price)+1) :
            return '第二类个性化设置中，位置选取超出价位列表长度'        
    
    if len(index_n) != len(min_prob):
        return "第三类个性化设置中，对应列表长度要一致"
    
    for l in range(len(index_n)) :
        if index_n[l] not in range(0,len(price)+1) :
            return '第三类个性化设置中，位置选取超出价位列表长度'
    
    #构造第一类个性化设置的参数列表
    a1 = []
    if index_i[0] != 0 and index_j[0] != 0:
        for l in range(len(index_i)) :
            aux_list = [0]*len(price)
            aux_list[index_i[l]-1] = -1
            aux_list[index_j[l]-1] = 1
            a1.append(aux_list)
     
    b1 = []
    if index_i[0] != 0 and index_j[0] != 0:
        b1 = [0]*len(index_i)
    
    #构造第二类个性化设置的参数列表
    
    a2 = []
    if index_m[0] != 0 :
        for l in range(len(index_m)):
            aux_list = [0]*len(price)
            aux_list[index_m[l]-1] = 1
            a1.append(aux_list)
            
    b2=[]        
    if index_m[0] != 0 :
        b2 = max_prob
        
    # 构造第三类个性化设置的参数列表
    a3 = []
    if index_n[0] != 0 :
        for l in range(len(index_n)):
            aux_list = [0]*len(price)
            aux_list[index_n[l]-1] = -1
            a1.append(aux_list)
            
    b3=[]        
    if index_n[0] != 0 :
        b3 = (-np.array(min_prob)).tolist()
    
    # 使用线性规划求解
    a_ub = np.array([price] + a1 + a2 + a3)
    b_ub = np.array([cost-profit] + b1 + b2 + b3)
        
    
    a_eq = np.array([[1]*len(price)])
    b_eq = np.array([1])
    bounds = [(e,None)]*len(price)
    res = linprog(z, A_ub= a_ub, b_ub = b_ub,A_eq=a_eq, b_eq = b_eq, bounds = bounds, method='interior-point')
    
    # 求出的解是否满足条件判断
    if cost - np.dot(price ,res.x) >= profit - 1 and abs(res.x.sum()-1) < 0.001:
        return dict(zip(price,[str(x) + "%" for x in np.around(res.x*100,decimals=3)]))
    else:
        return '参数设置不合理，根据当前价位，成本和预期利润设置，最低概率不能超过{:.3f}，'\
               '请调低最低概率或调高用户返利；若最低概率已满足条件，'\
               '则说明当前设置不能满足个性化需求'.format((cost-profit-min(price))/(np.sum(np.array(price)-min(price))))

# 转换输入格式
def convert_input(input_str, data_type):
    if data_type == 'int':
        return [int(x.strip()) for x in input_str.split(",")]
    elif data_type == 'float':
        return [float(x.strip()) for x in input_str.split(",")]

# 页面标题
st.markdown("## 商品抽奖概率自动生成工具")
# 输入参数
st.markdown("### 基本设置需求")
cost = st.number_input("用户抽奖花费", value=4000)
profit = st.number_input("平台期望利润利润", value=300)
price = st.text_area("价格分布", value="1200, 1500, 2000, 2500, 3000, 3500, 4000, 6000, 10000, 20000, 25000, 30000")
e = st.number_input("最低商品概率阈值", value=0.001, step=0.0001, format="%.4f")
st.markdown("### 个性化设置需求")
st.markdown("#### 1. 第 i 个位置商品概率不小于第 j 个位置商品概率")
index_i = st.text_input("位置列表 i", value="2,3,4,5,6,7")
index_j = st.text_input("位置列表 j", value="1,2,3,4,5,6")
st.markdown("#### 2. 第 m 个位置商品概率最高阈值")
index_m = st.text_input("位置列表 m ", value="1,2")
max_prob = st.text_input("最大概率列表", value="0.05,0.07")
st.markdown("#### 3. 第 n 个位置商品概率最低阈值")
index_n = st.text_input("位置列表 n", value="8,9")
min_prob = st.text_input("最小概率列表", value="0.1,0.05")

# 计算按钮
if st.button("计算"):
    try:
        # 调用计算函数
        result = calc_probability(
            e=e,
            price=convert_input(price, 'int'),
            cost=cost,
            profit=profit,
            index_i=convert_input(index_i, 'int'),
            index_j=convert_input(index_j, 'int'),
            index_m=convert_input(index_m, 'int'),
            max_prob=convert_input(max_prob, 'float'),
            index_n=convert_input(index_n, 'int'),
            min_prob=convert_input(min_prob, 'float')
        )     
        # 显示结果
        if isinstance(result, dict):
            st.success("计算结果：")
            df = pd.DataFrame(list(result.items()), columns=['价格', '概率'])
            # 显示表格
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.error(result)
    except Exception as e:
        st.error(f"计算出错: {str(e)}")
