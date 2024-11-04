import pandas as pd

# 加载数据集
# data = pd.read_csv('/home/yran1/NLP/proposal/all_the_data/HMS_2022-2023_PUBLIC_instchars.csv', low_memory=False)
data = pd.read_csv('/home/yran1/NLP/proposal/all_the_data/HMS_2021-2022_PUBLIC_instchars.csv', low_memory=False)

# 定义与抑郁和焦虑相关的列
depression_columns = ['dx_dep_1', 'dx_dep_2', 'dx_dep_3_new', 'dx_dep_4_new', 'dx_dep_4_text_new', 'dx_dep_5_new']
anxiety_columns = ['dx_ax_1', 'dx_ax_2', 'dx_ax_3', 'dx_ax_4', 'dx_ax_5', 'dx_ax_6_new', 'dx_ax_6_text_new', 'dx_ax_7_new']

# 创建抑郁和焦虑的二分类列
data['is_depressed'] = data[depression_columns].notna().any(axis=1).astype(int)
data['is_anxious'] = data[anxiety_columns].notna().any(axis=1).astype(int)

# 仅保留数值型列
data_filtered = data.drop(columns=depression_columns + anxiety_columns)
numeric_data = data_filtered.select_dtypes(include=['float64', 'int64'])

# # 计算相关性并排除自相关项和 is_anxious 与 is_depressed 之间的相关性
# correlation_with_depression = numeric_data.corrwith(data['is_depressed']).sort_values(ascending=False).drop(['is_depressed', 'is_anxious', 'meds_cur', 'meds_any', 'ther_ever', 'needmet_temp', 'ther_lifetime', 'gad7_1','gad7_2','gad7_3','gad7_4','gad7_5','gad7_6','gad7_7','phq2_1','phq2_2','phq9_1','phq9_2','phq9_3','phq9_4','phq9_5','phq9_6','phq9_7','phq9_8','phq9_9','ther_cur','ther_any','tx_any','dx_dep','dx_anx','dx_any'], errors='ignore')
# correlation_with_anxiety = numeric_data.corrwith(data['is_anxious']).sort_values(ascending=False).drop(['is_depressed', 'is_anxious', 'meds_cur', 'meds_any', 'ther_ever', 'needmet_temp', 'ther_lifetime', 'gad7_1','gad7_2','gad7_3','gad7_4','gad7_5','gad7_6','gad7_7','phq2_1','phq2_2','phq9_1','phq9_2','phq9_3','phq9_4','phq9_5','phq9_6','phq9_7','phq9_8','phq9_9','ther_cur','ther_any','tx_any','dx_dep','dx_anx','dx_any'], errors='ignore')
correlation_with_depression = numeric_data.corrwith(data['is_depressed']).sort_values(ascending=False).drop(['is_depressed', 'is_anxious', 'meds_cur','meds_any','needmet_temp','tx_any','ther_lifetime','phq2_1','phq2_2','phq9_1','phq9_2','phq9_3','phq9_4','phq9_5','phq9_6','phq9_7','phq9_8','phq9_9', 'dx_any', 'dx_dep', 'dx_anx','gad7_1','gad7_2','gad7_3','gad7_4','gad7_5','gad7_6','gad7_7'], errors='ignore')
correlation_with_anxiety = numeric_data.corrwith(data['is_anxious']).sort_values(ascending=False).drop(['is_depressed', 'is_anxious','meds_cur','meds_any','needmet_temp','tx_any','ther_lifetime','phq2_1','phq2_2','phq9_1','phq9_2','phq9_3','phq9_4','phq9_5','phq9_6','phq9_7','phq9_8','phq9_9', 'dx_any', 'dx_dep', 'dx_anx','gad7_1','gad7_2','gad7_3','gad7_4','gad7_5','gad7_6','gad7_7'], errors='ignore')

# 输出结果
print("与抑郁最相关的因素：")
print(correlation_with_depression.head(10))  # 输出前10个相关因素

print("\n与焦虑最相关的因素：")
print(correlation_with_anxiety.head(10))  # 输出前10个相关因素
