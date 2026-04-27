import json
import os
import shutil
from pathlib import Path

def classify_battery_data(json_dir, image_dir, output_dir):
    # 결과 저장을 위한 폴더 생성
    normal_path = Path(output_dir) / "normal"
    abnormal_path = Path(output_dir) / "abnormal"
    normal_path.mkdir(parents=True, exist_ok=True)
    abnormal_path.mkdir(parents=True, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    stats = {"normal": 0, "abnormal": 0}

    for json_file in json_files:
        with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 1. 기본 정보 추출
        image_name = data['image_info']['file_name']
        is_normal_flag = data['image_info']['is_normal'] # True/False
        defects = data.get('defects') # null 또는 결함 정보
        swelling = data.get('swelling', {}).get('swelling', False) # True/False

        # 2. 판별 로직: 하나라도 이상 징후가 있으면 비정상
        # - is_normal이 False이거나
        # - defects가 null이 아니거나
        # - swelling이 True인 경우
        if is_normal_flag is True and defects is None and swelling is False:
            status = "normal"
        else:
            status = "abnormal"

        # 3. 파일 복사 (이미지 파일이 존재할 경우)
        src_image = os.path.join(image_dir, image_name)
        if os.path.exists(src_image):
            dest_folder = normal_path if status == "normal" else abnormal_path
            shutil.copy(src_image, dest_folder / image_name)
            stats[status] += 1
            print(f"[{status.upper()}] {image_name} 분류 완료")
        else:
            print(f"[경고] 이미지를 찾을 수 없음: {image_name}")

    print("\n--- 분류 결과 요약 ---")
    print(f"정상 데이터: {stats['normal']}건")
    print(f"비정상 데이터: {stats['abnormal']}건")

if __name__ == "__main__":
    classify_battery_data(
        json_dir='raw_data/labels', 
        image_dir='raw_data/images', 
        output_dir='classified_dataset'
    )