import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

# ==========================================
# 1. 加载模型和配方
# ==========================================
try:
   model = joblib.load('model_lr.pkl')
   scaler = joblib.load('scaler_final.pkl')
   with open('selected_features.json', 'r') as f:
        features = json.load(f)
except Exception as e:
    st.error(f"⚠️ 模型加载失败，请检查文件。错误信息: {e}")
    st.stop()

# ==========================================
# 2. 网页基本设置
# ==========================================
st.set_page_config(page_title='老年人低质量睡眠风险预测', page_icon='🌙')
st.title('🌙 中国老年人低质量睡眠风险预测工具')
st.caption('基于 CHARLS 全国队列与机器学习模型（仅供学术研究，不替代医生诊断）')

# ==========================================
# 3. 侧边栏输入界面 (真正的用户友好版)
# ==========================================
st.sidebar.header('请输入老年人特征：')

age = st.sidebar.slider('年龄', 60, 100, 70)
gender = st.sidebar.selectbox('性别', ['男', '女'])
marital = st.sidebar.selectbox('婚姻状况', ['有配偶', '无配偶'])

# 学历下拉菜单：只显示中文，不显示恶心代码，后台自动匹配
edu_options = {
    '没上过学': 1,
    '小学': 4,
    '初中': 5,
    '高中/中专': 6,
    '大专及以上': 8
}
edu_selected = st.sidebar.selectbox('最高学历', list(edu_options.keys()))
education = edu_options[edu_selected]

bmi = st.sidebar.number_input('BMI (体重kg/身高m²)', 10.0, 40.0, 22.0, format="%.1f")

# 慢病勾选框
st.sidebar.markdown('**慢病情况（勾选符合的疾病）**')
hbp = st.sidebar.checkbox('高血压')
dm = st.sidebar.checkbox('糖尿病')
hd = st.sidebar.checkbox('心脏病')
stroke = st.sidebar.checkbox('中风')
chronic_count = sum([hbp, dm, hd, stroke])
st.sidebar.caption("注：基于模型的核心特征，纳入4种高发慢病。")

# 抑郁和午睡的友好提示
depression_score = st.sidebar.slider('抑郁得分 (CES-D)', 0, 30, 10, help="0-9分：正常；10-19分：轻度抑郁；20分以上：中重度抑郁。")
st.sidebar.caption("💡 参考：0-9分正常，10-19分轻度，20分以上中重度。")

da050 = st.sidebar.slider('午睡时长 (分钟)', 0, 180, 30, help="0代表不午睡，60代表1小时。")

da059 = st.sidebar.selectbox('是否吸烟', ['不吸', '吸'])
da067 = st.sidebar.selectbox('是否饮酒', ['不喝', '喝'])

# ==========================================
# 4. 数据处理与预测
# ==========================================
gender_val = 1 if gender == '男' else 0
marital_val = 1 if marital == '有配偶' else 0
da059_val = 1 if da059 == '吸' else 0
da067_val = 1 if da067 == '喝' else 0

# 组装数据
input_data = pd.DataFrame([[age, gender_val, marital_val, education, bmi, 
                            chronic_count, depression_score, da050, da059_val, da067_val]], 
                          columns=features)

# 标准化并预测
input_scaled = scaler.transform(input_data)
prob = model.predict_proba(input_scaled)[0][1]

# ==========================================
# 5. 结果展示
# ==========================================
st.markdown('---')
st.subheader('预测结果：')

if prob >= 0.5:
    st.error(f'⚠️ **高风险**：该老年人发生低质量睡眠的概率为 **{prob*100:.1f}%**。\n\n建议重点关注其睡眠健康，必要时提供心理或医疗干预。')
else:
    st.success(f'✅ **低风险**：该老年人发生低质量睡眠的概率为 **{prob*100:.1f}%**。\n\n当前睡眠状态相对良好，建议继续保持。')
