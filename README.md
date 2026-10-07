# psdl1 — DL1 제품 비교군 선정

신규 제품 하나의 도면을 넣어, unit 크기가 달라도 형상과 전극 연결이 닮은 기존 제품을 닮은 순서로 찾는다.
`particle-sim-compare`의 수치 계산과는 별도 저장소로 관리한다.

합성 자료로 코드가 도는 것까지만 확인했다. 실제 도면과 대응 정답은 폐쇄망에만 있으므로, 성능은 폐쇄망에서 학습해 봐야 안다.

## 하는 일과 하지 않는 일

- 한다: 제품 특징을 만들고 코사인 유사도로 순위를 낸다.
- 하지 않는다: 비교할 수 있는 불량 조합의 판정, CA·hotspot의 예측. 이 둘은 `particle-sim-compare` 쪽의 일이다.

## 입력

제품 하나는 전극 × layer 이미지들의 묶음이다. mask는 2-D `.npy`이고 0/255로 저장돼 있어도 0/1로 읽는다.

```json
{"products": [
  {"product_id": "A", "unit_size_um": [60, 60], "cnt": "A/cnt.npy",
   "elements": [
     {"role": "gate", "layer": "layer1", "mask": "A/gate_layer1.npy"},
     {"role": "data", "layer": "layer2", "mask": "A/data_layer2.npy"}
   ]}
]}
```

- `role`은 제품 간에 같은 역할을 뜻하는 이름이어야 한다. 제품마다 다른 임의 ID를 쓰면 안 된다.
- 경로는 manifest 폴더 기준 상대 경로이고, 폴더 밖을 가리키면 거부한다.
- 한 제품의 mask는 격자가 모두 같아야 한다. unit 하나만 담는다(2×2로 늘린 이미지가 아니다).

## 방식

1. **unit 전체를 같은 격자로 맞춘다.** 비율을 유지한 채 면적 평균으로 줄이고 남는 쪽은 빈칸으로 채운다. µm 기준 조각으로 자르지 않는 이유는, unit 크기가 다른 두 제품에서 조각 하나에 담기는 구조의 범위가 달라져 같은 배치가 다르게 보이기 때문이다.
2. **원소마다 3채널 이미지를 만든다.** 이 전극·layer / CNT / 제품의 나머지 배선.
3. **같은 인코더에 넣고 조건을 붙인다.** 역할과 layer를 임베딩해 이미지 특징과 이어 붙인 뒤 MLP로 합친다.
4. **Set Transformer로 제품 특징 하나를 만든다.** 원소 수가 달라도 되고 순서에 영향받지 않는다.
5. **학습**: 같은 제품을 다르게 변형한 두 장은 가깝게, 다른 제품은 구분한다. 변형은 제품의 모든 원소에 똑같이 준다(무작위로 일부를 잘라 키우고 조금 옮김).

절대 치수는 이 모델이 보지 않는다. 간격이 몇 µm인지는 DL2와 수치 계산이 맡는다.

## 설치

```bash
uv venv && uv pip install -e . pytest
uv pip install timm        # ConvNeXt V2 등 timm 모델을 쓸 때만
```

## 학습과 검색

```bash
# 내장 작은 인코더로 시험
python -m psdl1.train --manifest data/manifest.json --out runs/dl1 --epochs 200 --grid 256

# ConvNeXt V2, 처음부터 학습
python -m psdl1.train --manifest data/manifest.json --out runs/dl1 --backbone convnextv2_tiny

# 가중치 파일에서 시작 (네트워크로 받지 않는다)
python -m psdl1.train ... --backbone convnextv2_tiny --weights weights/convnextv2_tiny.safetensors

# 제품 대응 정답이 있을 때
python -m psdl1.train ... --relations data/relations.csv

# 검색
python -m psdl1.search --ckpt runs/dl1/dl1.pt --manifest data/manifest.json --query B --topk 5
```

`relations.csv`는 머리글 없이 `제품A,제품B,similar` 또는 `제품A,제품B,different` 한 줄씩이다.

- 정답 파일이 없으면 다른 제품은 모두 구분할 대상으로 본다.
- 정답 파일이 있으면 적힌 쌍만 쓰고, 적히지 않은 쌍은 손실 계산에서 뺀다.

## 폐쇄망에서 정해야 할 것

- **격자 크기(`--grid`)**: 가장 큰 unit을 줄였을 때 가장 가는 선이 2픽셀 이상 남아야 한다. `psdl1.data.narrowest_run_px`로 원본의 가장 가는 선을 재서 `격자 ÷ unit 픽셀 수`를 곱해 본다.
- **정답 없이 학습할 때의 한계**: 실제로 닮은 제품끼리도 서로 밀어내게 된다. 닮은 제품이 많으면 정답 파일을 주는 편이 낫다.
- **unit 크기 조건(`--use-unit-size`)**: 넣은 경우와 뺀 경우를 비교한다. 넣으면 크기가 다른 제품을 덜 닮았다고 볼 수 있다.
- **평가**: unit 크기가 비슷한 제품끼리만 가깝게 나오는지, 크기가 다른데 구조가 닮은 제품을 찾아내는지를 따로 본다. 단순 이미지 비교보다 나은지도 확인한다.
- **건너뛴 layer**: 입력은 원본 형상이다. 사이 layer의 이미지를 함께 넣을지는 미정이다.

## 사전학습 가중치

- 가중치는 파일에서만 읽는다. `timm`의 `pretrained_cfg_overlay={"file": ...}`를 쓰므로 네트워크 연결이 필요 없다.
- **라이선스를 확인해야 한다.** `timm`의 코드는 Apache-2.0이지만, 공개된 ConvNeXt V2 가중치는 비상업적 이용 조건(CC BY-NC 4.0)으로 배포된 것으로 알고 있다. 사내 사용 가능 여부를 반입 전에 확인한다. 확인 전에는 가중치 없이 처음부터 학습하는 경로를 쓴다.
- 도면 mask는 사진과 성격이 달라, 사진으로 학습한 가중치가 도움이 되는지는 비교해 봐야 한다.

## 아직 없는 것

- FCMAE 방식의 사전학습(가리고 복원하기). 지금은 대조 학습만 있다.
- 전극 사이의 연결 정보를 그래프로 직접 넣는 것. 지금은 CNT 채널과 나머지 배선 채널로만 들어간다.

## 테스트

```bash
python -m pytest
```
