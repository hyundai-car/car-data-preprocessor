from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

import json
import pickle
import traceback

import catboost
import numpy as np
import pandas as pd
import re
import math
from typing import Dict, Any
from datetime import datetime

# 데이터 모델 정의
class AccidentHistory(BaseModel):
    parts: float = Field(..., description="부품 비용")
    labor: float = Field(..., description="공임 비용")
    painting: float = Field(..., description="도장 비용")

class CarAnalysisRequest(BaseModel):
    carName: str = Field(..., description="차량 이름")
    carType: str = Field(..., description="차량 타입")
    carYear: str = Field(..., description="차량 연식 (YYYY.MM 형식)")
    mileage: float = Field(..., description="주행거리")
    displacement: float = Field(..., description="배기량")
    sellingPrice: float = Field(..., description="판매가")
    fuelType: str = Field(..., description="연료 타입")
    newCarPrice: float = Field(..., description="신차가")
    accidentHistoryList: List[AccidentHistory] = Field(default_factory=list, description="사고이력 리스트")

class CarAnalysisResponse(BaseModel):
    mmScore: Optional[float] = Field(None, description="MM 스코어")
    accidentSeverity: float = Field(..., description="사고 심각도")
    repairProbability: Optional[float] = Field(None, description="수리 확률")
    predictedPrice: float = Field(..., description="예측 가격")
    cityEfficiency: Optional[float] = Field(None, description="도시 연비")
    highwayEfficiency: Optional[float] = Field(None, description="고속도로 연비")
    input_received: CarAnalysisRequest = Field(..., description="입력받은 데이터")

# FastAPI 앱 초기화
app = FastAPI(
    title="Car Analysis API",
    description="차량 분석 API",
    version="1.0.0"
)

# 전역 변수
model = None
age_data = None
fuel_data = None
mileage_data = None
efficiency_df = None

def init_data():
    """데이터 초기화 함수"""
    global model, age_data, fuel_data, mileage_data, efficiency_df
    if model is None:
        with open('./data/car_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
    if age_data is None:
        age_data = pd.read_csv('./data/car_inspection_by_age_20231231.csv')
        fuel_data = pd.read_csv('./data/car_inspection_by_fuel_type_20240807.csv')
        mileage_data = pd.read_csv('./data/car_inspection_by_mileage_20240807.csv')
        efficiency_df = pd.read_csv('./data/hyundai_cars_fuel_efficiency.csv')

def get_inspection_car_type(car_type):
    """차종을 검사 데이터 차종으로 변환"""
    if car_type in ['passenger', 'luxury']:
        return '승용차'
    elif car_type in ['van', 'suv']:
        return '승합차'
    elif car_type == 'cargo':
        return '화물차'
    return None

def get_failure_rate_by_age(age_data, car_type, car_age):
    """연령별 부적합률 계산"""
    inspection_car_type = get_inspection_car_type(car_type)
    if not inspection_car_type:
        return None

    age_mapping = {
        (0, 48): ('2023년 4년 이하 부적합률 (퍼센트)', '2023년 4년 이하 검사대수 (대)'),
        (49, 72): ('2023년 5-6년 부적합률 (퍼센트)', '2023년 5-6년 검사대수 (대)'),
        (73, 96): ('2023년 7-8년 부적합률 (퍼센트)', '2023년 7-8년 검사대수 (대)'),
        (97, 120): ('2023년 9-10년 부적합률 (퍼센트)', '2023년 9-10년 검사대수 (대)'),
        (121, 144): ('2023년 11-12년 부적합률 (퍼센트)', '2023년 11-12년 검사대수 (대)'),
        (145, 168): ('2023년 13-14년 부적합률 (퍼센트)', '2023년 13-14년 검사대수 (대)'),
        (169, float('inf')): ('2023년 15년 이상 부적합률 (퍼센트)', '2023년 15년 이상 검사대수 (대)')
    }

    for (min_age, max_age), (failure_col, count_col) in age_mapping.items():
        if min_age <= car_age <= max_age:
            break

    car_data = age_data[age_data['차종별'] == inspection_car_type]
    if car_data.empty:
        return None

    failure_rates = car_data[failure_col]
    inspection_counts = car_data[count_col]

    total_count = inspection_counts.sum()
    if total_count == 0:
        return None

    weighted_failure_rate = (failure_rates * inspection_counts).sum() / total_count
    return weighted_failure_rate

def get_failure_rate_by_fuel(fuel_data, car_type, fuel_type):
    """연료별 부적합률 계산"""
    inspection_car_type = get_inspection_car_type(car_type)
    if not inspection_car_type:
        return None

    fuel_mapping = {
        'gasoline': ('휘발유부적합률 (퍼센트)', '휘발유검사대수 (대)'),
        'diesel': ('경유부적합률 (퍼센트)', '경유검사대수 (대)'),
        'hybrid': ('하이브리드부적합률 (퍼센트)', '하이브리드검사대수 (대)'),
        'electric': ('전기부적합률 (퍼센트)', '전기검사대수 (대)'),
        'other': ('기타부적합률 (퍼센트)', '기타검사대수 (대)')
    }

    failure_col, count_col = fuel_mapping.get(fuel_type, (None, None))
    if not failure_col or not count_col:
        return None

    car_data = fuel_data[fuel_data['차종별'] == inspection_car_type]
    if car_data.empty:
        return None

    failure_rates = car_data[failure_col].str.rstrip('%').astype('float')
    inspection_counts = car_data[count_col]

    total_count = inspection_counts.sum()
    if total_count == 0:
        return None

    weighted_failure_rate = (failure_rates * inspection_counts).sum() / total_count
    return weighted_failure_rate

def get_failure_rate_by_mileage(mileage_data, car_type, mileage):
    """주행거리별 부적합률 계산"""
    inspection_car_type = get_inspection_car_type(car_type)
    if not inspection_car_type:
        return None

    mileage_mapping = {
        (0, 50000): ('5만킬로미터 미만부적합률 (퍼센트)', '5만킬로미터미만검사대수 (대)'),
        (50000, 100000): ('5만010만킬로미터 미만부적합률 (퍼센트)', '5만010만킬로미터 미만검사대수 (대)'),
        (100000, 150000): ('10만015만킬로미터 미만부적합률 (퍼센트)', '10만015만킬로미터 미만검사대수 (대)'),
        (150000, 200000): ('15만020만킬로미터 미만부적합률 (퍼센트)', '15만020만킬로미터 미만검사대수 (대)'),
        (200000, 250000): ('20만025만킬로미터 미만부적합률 (퍼센트)', '20만025만킬로미터 미만검사대수 (대)'),
        (250000, 300000): ('25만030만킬로미터 미만부적합률 (퍼센트)', '25만030만킬로미터 미만검사대수 (대)'),
        (300000, 350000): ('30만035만킬로미터 미만부적합률 (퍼센트)', '30만035만킬로미터 미만검사대수 (대)'),
        (350000, float('inf')): ('35만킬로미터 이상부적합률 (퍼센트)', '35만킬로미터 이상검사대수 (대)')
    }

    for (min_mile, max_mile), (failure_col, count_col) in mileage_mapping.items():
        if min_mile <= mileage < max_mile:
            break

    car_data = mileage_data[mileage_data['차종별'] == inspection_car_type]
    if car_data.empty:
        return None

    failure_rates = car_data[failure_col].str.rstrip('%').astype('float')
    inspection_counts = car_data[count_col]

    total_count = inspection_counts.sum()
    if total_count == 0:
        return None

    weighted_failure_rate = (failure_rates * inspection_counts).sum() / total_count
    return weighted_failure_rate

def calculate_repair_probability(age_data, fuel_data, mileage_data, car_type, car_age,
    fuel_type, mileage):
    """repair_probability 계산 (기하평균 사용)"""
    age_failure = get_failure_rate_by_age(age_data, car_type, car_age)
    fuel_failure = get_failure_rate_by_fuel(fuel_data, car_type, fuel_type)
    mileage_failure = get_failure_rate_by_mileage(mileage_data, car_type, mileage)

    if any(x is None for x in [age_failure, fuel_failure, mileage_failure]):
        return None

    health_scores = [
        age_failure,
        fuel_failure,
        mileage_failure
    ]

    geometric_mean = np.prod(health_scores) ** (1 / 3)
    return geometric_mean

def calculate_value_score(predicted_price, selling_price, new_car_price, is_genesis=False):
    MAPE = 7.19

    error = ((selling_price - predicted_price) / selling_price) * 100

    if is_genesis:
        k = -math.log(0.7) / (4 * MAPE * MAPE)
        if error <= -MAPE:
            score = 100 * math.exp(-(3 * k) * (error + MAPE) ** 2)
        else:
            score = 100 * math.exp(-k * (error + MAPE) ** 2)
    else:
        k = -math.log(0.5) / (4 * MAPE * MAPE)
        if error <= -MAPE:
            score = 100 * math.exp(-(3 * k) * (error + MAPE) ** 2)
        else:
            score = 100 * math.exp(-k * (error + MAPE) ** 2)

    return score

def extract_fuel_type(model_name):
    """연비 데이터의 model 컬럼에서 연료 타입 추출"""
    model_name = model_name.lower()

    if '하이브리드' in model_name or 'hybrid' in model_name:
        return 'hybrid'
    elif '디젤' in model_name or 'diesel' in model_name:
        return 'diesel'
    elif '가솔린' in model_name or 'gasoline' in model_name:
        if '전기' in model_name:
            return 'hybrid'
        return 'gasoline'
    elif '전기' in model_name or 'electri' in model_name or '아이오닉' in model_name:
        return 'electric'

    return 'gasoline'

def find_fuel_efficiency(car_data, efficiency_df):
    try:
        efficiency_df['extracted_fuel_type'] = efficiency_df['model'].apply(extract_fuel_type)

        filtered_df = efficiency_df[
            efficiency_df['extracted_fuel_type'] == car_data['fuel_type']
            ].copy()

        if len(filtered_df) == 0:
            return None, None, None

        base_models = ['그랜저', '쏘나타', '아반떼', '투싼', '싼타페', '팰리세이드', '베뉴', '코나', '캐스퍼', '아이오닉6', '스타리아',
                       'G70', 'GV70', 'G80', 'GV80', 'G90', 'GV90']

        source_name = re.sub(r'^\d{4}\s*', '', str(car_data['car_name'])).lower()
        source_base = next((model for model in base_models if model.lower() in source_name.lower()), None)

        if not source_base:
            return None, None, None

        filtered_df = filtered_df[
            filtered_df['model'].str.lower().apply(lambda x: source_base.lower() in x)
        ].copy()

        if len(filtered_df) == 0:
            return None, None, None

        def calculate_similarity(row):
            target_name = str(row['model']).lower()
            similarity = 0.6

            if car_data['displacement'] > 0:
                try:
                    target_displacement = pd.to_numeric(row['displacement'].replace(',', ''), errors='coerce')
                    if target_displacement == car_data['displacement']:
                        similarity += 0.2
                except:
                    pass

            keywords = ['IG', 'HG', 'CN7', 'NF', '터보', 'turbo', '인치']

            for keyword in keywords:
                if (keyword.lower() in source_name) == (keyword.lower() in target_name):
                    similarity += 0.05

            return min(similarity, 1.0)

        filtered_df.loc[:, 'similarity'] = filtered_df.apply(calculate_similarity, axis=1)
        matches = filtered_df[filtered_df['similarity'] >= 0.6].copy()

        if len(matches) > 0:
            best_match = matches.loc[matches['similarity'].idxmax()]
            return (
                best_match['model'],
                float(best_match['city_efficiency']),
                float(best_match['highway_efficiency'])
            )

        return None, None, None

    except Exception as e:
        print(f"연비 정보 매칭 중 오류 발생: {e}")
        return None, None, None

def calculate_car_age(year_str):
    """차량 연식 계산"""
    current_date = datetime.now()
    year_str, month_str, *_ = year_str.split('.')
    year = int(year_str)
    month = int(month_str)
    car_date = year * 12 + month
    current_date = current_date.year * 12 + current_date.month
    return current_date - car_date

def analyze_repair_severity(costs, new_car_price):
    """단일 사고의 심각도 분석"""
    MAX_REPAIR = new_car_price * 10000

    total = costs['total']
    if total == 0:
        return 0

    log_ratio = np.log10(total + 1) / np.log10(MAX_REPAIR + 1)
    base_severity = log_ratio * 85

    structure_score = 0

    if costs['부품'] > 0:
        parts_ratio = costs['부품'] / total
        structure_score += parts_ratio * 10

    painting_labor_ratio = costs['도장'] / (costs['공임'] + 1e-6)

    if painting_labor_ratio > 1.2:
        structure_score += 2
    elif painting_labor_ratio < 0.8:
        structure_score += 3
    else:
        structure_score += 5

    final_severity = base_severity + structure_score

    if final_severity <= 20:
        final_severity = 5 + (final_severity / 20) * 5

    return min(round(final_severity, 2), 100)


def analyze_multiple_accidents(accident_list, new_car_price):
    """여러 사고의 종합 심각도 분석"""
    if not accident_list:
        return 0

    severities = []
    for accident in accident_list:
        costs = {
            '부품': float(accident.get('parts', 0)),
            '공임': float(accident.get('labor', 0)),
            '도장': float(accident.get('painting', 0)),
            'total': float(accident.get('parts', 0)) + float(accident.get('labor', 0)) + float(
                accident.get('painting', 0))
        }
        severity = analyze_repair_severity(costs, new_car_price)
        severities.append(severity)

    if len(severities) == 1:
        return severities[0]

    severities.sort(reverse=True)
    final_severity = severities[0]

    for i in range(1, len(severities)):
        impact = severities[i] * (0.7 ** i)
        remaining_severity = 100 - final_severity
        if remaining_severity > 0:
            final_severity += impact * (remaining_severity / 100)

    return round(min(final_severity, 100), 2)


@app.on_event("startup")
async def startup_event():
    """앱 시작시 데이터 초기화"""
    init_data()

@app.post("/analyze", response_model=CarAnalysisResponse)
async def analyze_car(request: CarAnalysisRequest):
    try:
        # 차량 연식 계산
        car_age = calculate_car_age(request.carYear)

        # 사고 심각도 계산
        accident_severity = analyze_multiple_accidents(
            [accident.dict() for accident in request.accidentHistoryList],
            request.newCarPrice
        )

        # repair_probability 계산
        repair_probability = calculate_repair_probability(
            age_data, fuel_data, mileage_data,
            request.carType, car_age, request.fuelType, request.mileage
        )

        # 가격 예측
        input_features = [
            request.mileage, request.carType, request.fuelType, car_age,
            repair_probability, request.displacement, request.newCarPrice,
            accident_severity
        ]
        predicted_price = float(model.predict([input_features])[0])

        # mm_score 계산
        mm_score = calculate_value_score(
            predicted_price, request.sellingPrice, request.newCarPrice,
            is_genesis=(request.carType == 'luxury')
        )

        # 연비 정보 찾기
        car_data = {
            'car_name': request.carName,
            'fuel_type': request.fuelType,
            'displacement': request.displacement
        }
        _, city_efficiency, highway_efficiency = find_fuel_efficiency(car_data, efficiency_df)

        # 응답 데이터 구성
        response = CarAnalysisResponse(
            mmScore=round(float(mm_score), 2) if mm_score is not None else None,
            accidentSeverity=round(accident_severity, 2),
            repairProbability=round(float(repair_probability), 2) if repair_probability is not None else None,
            predictedPrice=round(predicted_price, 2),
            cityEfficiency=round(float(city_efficiency), 2) if city_efficiency is not None else None,
            highwayEfficiency=round(float(highway_efficiency), 2) if highway_efficiency is not None else None,
            input_received=request
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'error_type': type(e).__name__,
                'error_detail': traceback.format_exc()
            }
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)