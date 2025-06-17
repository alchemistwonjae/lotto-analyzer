# app.py 최종 완성 코드 (컬럼 위치 수정 완료)
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
import gdown
import os
import random
from collections import Counter

# --- 데이터 로드 및 전처리 (최종 수정 버전) ---
@st.cache_data(ttl="1d")
def download_and_load_data(url, output_path="lotto_downloaded.xlsx"):
    st.info(f"Google Drive에서 엑셀 파일을 다운로드합니다...")
    gdown.download(url, output_path, quiet=False, fuzzy=True)
    
    st.info(f"다운로드한 {output_path} 파일을 읽습니다...")
    try:
        # 첫 번째 줄을 헤더(컬럼 이름)로 읽기
        df = pd.read_excel(output_path, header=0) 
        st.success("데이터를 성공적으로 가져왔습니다.")
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 데 실패했습니다: {e}")
        return None

    st.info("데이터 클리닝을 시작합니다...")
    original_rows = len(df)
    
    # ------------------- 여기가 최종 수정된 핵심 부분! -------------------
    # 실제 로또 번호가 있는 컬럼 위치를 2번부터 8번 전까지(3번째~8번째 칸)로 정확히 지정
    lotto_cols = df.columns[2:8]
    # ------------------------------------------------------------------
    
    for col in lotto_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=lotto_cols, inplace=True)
    
    for col in lotto_cols:
        df = df[df[col].between(1, 45)]

    for col in lotto_cols:
        df[col] = df[col].astype(int)
    
    try:
        df[df.columns[0]] = df[df.columns[0]].astype(int)
    except (ValueError, TypeError):
        pass

    cleaned_rows = len(df)
    st.info(f"클리닝 완료: {original_rows - cleaned_rows}개의 유효하지 않은 행을 제거했습니다. 현재 {cleaned_rows}개의 데이터로 분석합니다.")

    if df.empty:
        st.error("데이터 클리닝 후 남은 유효한 데이터가 없습니다. 원본 엑셀 파일의 내용을 확인해주세요.")
        return None

    df = df.sort_values(by=df.columns[0], ascending=True)
    return df

# --- (이 아래 코드는 이전과 동일합니다) ---

def create_sequence(data, sequence_length=10):
    x_data, y_data = [], []
    for i in range(len(data) - sequence_length):
        x_data.append(data[i:i + sequence_length])
        y_data.append(data[i + sequence_length])
    return np.array(x_data), np.array(y_data)

@st.cache_resource
def create_model(model_type, x, y):
    if model_type == 'lstm':
        model = Sequential([LSTM(128, input_shape=(x.shape[1], x.shape[2]), return_sequences=False), Dropout(0.2), Dense(64, activation='relu'), Dense(6)])
    else: # gru
        model = Sequential([GRU(128, input_shape=(x.shape[1], x.shape[2]), return_sequences=False), Dropout(0.2), Dense(64, activation='relu'), Dense(6)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=50, batch_size=32, verbose=0)
    return model

def get_prediction_from_model(model, sequence):
    predicted_normalized = model.predict(sequence)
    predicted_numbers = np.clip(predicted_normalized * 45, 1, 45).flatten()
    final_numbers = set()
    sorted_preds = np.argsort(-predicted_numbers)
    for idx in sorted_preds:
        num = round(predicted_numbers[idx])
        if 1 <= num <= 45: final_numbers.add(int(num))
        if len(final_numbers) == 6: break
    while len(final_numbers) < 6:
        new_num = random.randint(1, 45)
        if new_num not in final_numbers: final_numbers.add(new_num)
    return sorted(list(final_numbers))

def get_pairing_numbers(df):
    lotto_numbers = df.iloc[:, 2:8].values
    pair_counts = Counter()
    for row in lotto_numbers:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                pair = tuple(sorted((row[i], row[j])))
                pair_counts[pair] += 1
    base_num = random.choice(lotto_numbers[-1])
    partners = Counter()
    for pair, count in pair_counts.items():
        if base_num in pair:
            other = pair[0] if pair[1] == base_num else pair[1]
            partners[other] += count
    result = {base_num}
    top_partners = partners.most_common(5)
    for partner, count in top_partners:
        result.add(int(partner))
    while len(result) < 6:
        new_num = random.randint(1, 45)
        if new_num not in result: result.add(new_num)
    return sorted(list(result))

def get_consecutive_numbers():
    numbers = set()
    start_num = random.randint(1, 44)
    numbers.add(start_num)
    numbers.add(start_num + 1)
    while len(numbers) < 6:
        new_num = random.randint(1, 45)
        if new_num not in numbers: numbers.add(new_num)
    return sorted(list(numbers))

def get_random_numbers():
    return sorted(list(np.random.choice(range(1, 46), 6, replace=False)))

st.set_page_config(page_title="🤖 로또 전략 분석기", page_icon="🍀")
st.title("AI 로또 전략 분석기")
st.write("5가지 서로 다른 전략으로 행운의 번호 조합을 생성합니다.")

GOOGLE_DRIVE_EXCEL_URL = "https://docs.google.com/spreadsheets/d/1zozJbbHc8GeC31M-3t5d1br2ZPSV9wL1/edit?usp=sharing&ouid=105102746496731688805&rtpof=true&sd=true"

try:
    df = download_and_load_data(GOOGLE_DRIVE_EXCEL_URL)

    if df is not None:
        lotto_numbers_normalized = df.iloc[:, 2:8].values / 45.0
        SEQUENCE_LENGTH = 10
        if len(lotto_numbers_normalized) > SEQUENCE_LENGTH:
            x, y = create_sequence(lotto_numbers_normalized, SEQUENCE_LENGTH)
            latest_sequence = lotto_numbers_normalized[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 6)
            st.success("모든 데이터와 전략이 준비되었습니다!")
            if st.button("5가지 전략으로 번호 생성하기!"):
                with st.spinner('AI와 패턴 전문가들이 총출동하여 번호를 분석하는 중...🤖'):
                    lstm_model = create_model('lstm', x, y)
                    gru_model = create_model('gru', x, y)
                    all_predictions = {
                        "딥러닝 LSTM 예측 (AI의 선택)": get_prediction_from_model(lstm_model, latest_sequence),
                        "궁합수 패턴 분석": get_pairing_numbers(df),
                        "연속 번호 조합": get_consecutive_numbers(),
                        "다른 딥러닝 GRU 예측": get_prediction_from_model(gru_model, latest_sequence),
                        "완전 무작위 자동 선택 (Quick Pick)": get_random_numbers()
                    }
                st.success("5가지 전략 분석이 완료되었습니다!")
                st.balloons()
                for strategy, numbers in all_predictions.items():
                    st.markdown(f"##### {strategy}")
                    st.markdown(f"""
                    <div style="display: flex; gap: 10px; font-size: 1.2rem; margin-bottom: 20px;">
                        {''.join(f'<div style="background-color: #007bff; color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-weight: bold;">{num}</div>' for num in numbers)}
                    </div>""", unsafe_allow_html=True)
        else:
            st.warning(f"딥러닝 모델을 학습시키기에 데이터가 부족합니다. 최소 {SEQUENCE_LENGTH + 1}개의 유효한 데이터 행이 필요합니다.")
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")