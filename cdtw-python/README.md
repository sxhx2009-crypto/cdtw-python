# CDTW Boundary-Sampling for Python

두 수열을 **1차원 다각선 곡선**으로 해석하고, 호 길이(arc length)로
매개화한 뒤 Continuous Dynamic Time Warping을 계산하는 NumPy 구현입니다.

이 저장소는 `pyproject.toml` 기반으로 설치할 수 있으며, push와 pull request마다
GitHub Actions가 Python 3.10~3.13 단위 테스트와 패키지 빌드를 자동 검사합니다.

## GitHub에서 바로 설치

저장소를 올린 뒤 `YOUR_ID`를 실제 GitHub 사용자 이름으로 바꿔 실행합니다.

```bash
py -m pip install "git+https://github.com/YOUR_ID/cdtw-python.git@v0.2.0"
```

아직 태그를 만들지 않았다면 `@v0.2.0` 대신 `@main`을 사용할 수 있습니다.
자세한 최초 업로드 방법은 [`GITHUB_UPLOAD_GUIDE_KO.md`](GITHUB_UPLOAD_GUIDE_KO.md)에
정리되어 있습니다.

## 중요: 원래 시간축은 입력에 포함되지 않습니다

입력 배열의 인덱스나 실제 측정 시각은 좌표로 사용되지 않습니다. 값들을 잇는
1차원 곡선을 값 공간의 호 길이로 다시 매개화하므로, 연속 중복값과 원래 표본
간격 정보는 사라집니다. 특히 두 입력이 모두 상수 곡선이면 값이 서로 달라도
매개공간 길이가 0이라 결과가 `0.0`입니다.

따라서 이 코드는 논문의 1차원 CDTW 정의를 실험하기 위한 구현이며, 실제 시간
간격 자체를 보존해야 하는 분석에는 그대로 사용하면 안 됩니다.

## 구현 범위

논문에서 사용하는 CDTW는 매개공간의 단조 경로 `gamma`에 대해 다음 선적분을
최소화합니다.

```text
min_gamma integral |P(x) - Q(y)| ds_L1
```

이 코드는 각 원래 선분 쌍이 만드는 셀의 경계를 표본화하고 동적 계획법으로
단조 경로를 찾습니다. 셀 안에서 두 경계점 사이의 최적 연속 경로 비용은
해석적인 닫힌형 공식으로 정확히 계산합니다. 따라서 오차는 **셀 내부 적분**이
아니라 **경로가 셀 경계를 통과할 수 있는 위치를 유한하게 표본화한 것**에서만
생깁니다. 두 입력이 각각 하나의 선분이면 셀 경계를 중간에 통과할 필요가
없으므로 격자 크기와 무관하게 정확한 값을 냅니다.

중요: 이 구현은 실용적인 **수렴 격자 근사**이며, Buchin–Nusser–Wong 논문의
조각별 이차 경계함수 전파를 사용하는 정확한 `O((n+m)^5)` 알고리즘은 아닙니다.
`cdtw_adaptive`가 반환하는 `estimated_error`도 연속 최적값에 대한 보증된 오차
상한이 아니라, 최근 안정화 창에서 관측된 해상도 간 변화량의 최댓값입니다.

## 필요 패키지

```bash
python -m pip install numpy
```

저장소를 내려받아 수정하면서 사용하려면 다음처럼 editable 설치합니다.

```bash
git clone https://github.com/YOUR_ID/cdtw-python.git
cd cdtw-python
py -m venv .venv
.venv\Scripts\activate
py -m pip install -e ".[dev]"
```

## 빠른 사용법

```python
from cdtw import cdtw_distance

p = [0.0, 1.0, 0.2, 1.5, 1.0]
q = [0.0, 0.8, 0.4, 1.4, 1.0]

distance = cdtw_distance(p, q, grid_size=256)
print(distance)
```

정합 경로와 계산 정보를 함께 받으려면 다음처럼 사용합니다.

```python
from cdtw import cdtw

result = cdtw(p, q, grid_size=256, return_path=True)
print(result.distance)
print(result.parameter_path)  # 각 행은 호 길이 좌표 (x, y)
print(result.value_path)      # 각 행은 실제로 대응된 값 (P(x), Q(y))
```

해상도를 자동으로 두 배씩 늘려 결과 변화량을 확인하려면:

```python
from cdtw import cdtw_adaptive

result = cdtw_adaptive(
    p,
    q,
    initial_grid_size=32,
    max_grid_size=1024,
    rtol=1e-4,
    atol=1e-8,
    convergence_checks=3,
    return_path=True,
)

print(result.distance)
print(result.converged)
print(result.estimated_error)
print(result.history)
```

`convergence_checks=3`은 마지막 세 번의 해상도 변화가 모두 허용오차 안에 들어와야
멈추게 합니다. 한 단계 동안 값이 우연히 같았다가 다음 단계에서 다시 감소하는
평탄 구간 때문에 조기 종료되는 문제를 줄입니다. 그래도 이는 엄밀한 연속 최적값
오차 보증이 아니라 경험적인 안정화 판정입니다.

논문의 원래 선적분값이 기본값입니다. 서로 다른 곡선 길이의 영향을 줄인 평균
비용이 필요하면 `normalized=True`를 지정할 수 있습니다.

## 테스트 실행

저장소 루트에서 다음 명령을 실행합니다.

```bash
python -m unittest discover -s tests -v
python examples/basic_usage.py
python validation/validation_suite.py
```

테스트에는 동일 곡선의 거리 0, 대칭성, 닫힌형 해를 아는 반대 방향 선분,
퇴화한 점-선분 경우, 그리고 아래에 설명한 공선점 재표본화 편차와 극단적인
크기비 입력에 대한 회귀 검사가 포함됩니다. `validation_suite.py`는
고정밀 해석해, 수백 건의 변환 불변성 검사, 반환 경로 독립 재적분, 별도
격자-DP와의 차등검사, adaptive 평탄 구간 회귀, 메모리 가드, 세분화 단조성 및
스트레스 시험까지 수행합니다.
이번 검증의 방법과 수치는 `validation/VALIDATION_REPORT.md`와
`validation/validation_results.json`에 정리되어 있습니다.

## 저장소 구조

```text
cdtw.py                         핵심 공개 모듈
pyproject.toml                  pip 빌드·설치 설정
tests/test_cdtw.py              단위 및 회귀 테스트
examples/basic_usage.py         실행 예제
validation/validation_suite.py  전문 검증 스위트
validation/VALIDATION_REPORT.md 검증 보고서
.github/workflows/tests.yml     자동 CI 테스트
```

## 라이선스

권리 조건을 사용자가 직접 선택할 수 있도록 아직 `LICENSE` 파일을 포함하지
않았습니다. 저장소를 공개하고 다른 사람의 복제·수정·배포를 허용하려면
`GITHUB_UPLOAD_GUIDE_KO.md`의 안내에 따라 공개 전에 라이선스를 선택하세요.

## 계산량과 주의점

- 메모리는 생성된 두 매개변수 격자 크기의 곱에 비례합니다.
- 기본 `memory_limit_mib=512.0`은 전역 DP 배열과 가장 큰 셀의 임시행렬을
  보수적으로 추정해 한도를 넘으면 실제 대형 할당 전에 `MemoryError`를 냅니다.
  시스템 메모리를 확인한 경우에만 한도를 높이거나 `None`으로 해제하세요.
- `return_path=True`는 누적 비용 배열 외에 같은 크기의 `int64` 선행자 배열을
  4개 더 사용하므로 전역 DP 저장공간이 약 5배가 됩니다.
- 실행시간은 각 원래 셀에서 표본화된 입력·출력 경계점 수의 곱을 모두 더한
  값에 비례하므로, 곡선의 선분 길이 분포에도 영향을 받습니다.
- `grid_size`를 2배로 늘리면 일반적으로 시간과 메모리가 대략 4배 규모로
  증가할 수 있습니다.
- 각 곡선의 시작점과 끝점에서 대칭인 경계 표본을 사용하므로, 두 곡선을 함께
  역순으로 뒤집어도 유한 격자 결과가 동일합니다(부동소수점 반올림 범위 내).
- **공선점 재표본화는 유한 격자에서 불변이 아닙니다.** 곡선 모양을 바꾸지 않는
  꼭짓점을 끼워 넣어도 셀이 쪼개지고, 경로는 셀 경계를 표본점에서만 통과할 수
  있으므로 값이 **올라갈 수 있습니다**. 무작위 200쌍에서 `grid_size=128` 기준
  최대 `+4.1e-3`(상대)까지 관측됐고, 격자를 키우면 사라집니다(`2048`에서
  `+4.0e-6`). 즉 재표본화 불변성은 점근적으로만 성립합니다.
- **삼각부등식은 성립하지 않습니다.** 원시 선적분은 거리 함수(metric)가
  아닙니다. 무작위 200개 삼중쌍에서 21건이 `d(a,c) > d(a,b)+d(b,c)`를 위반했고,
  이는 구현 결함이 아니라 CDTW 정의 자체의 성질입니다.
- 입력의 연속 중복값은 길이 0인 선분이므로 제거됩니다.
- 곡선 꼭짓점의 호 길이 좌표는 격자에서 **정확히 보존**됩니다. 전체 호 길이
  대비 `eps` 미만인 선분이나 총 길이가 `1e-13` 이하인 곡선도 여기에 포함됩니다.
- 두 입력 모두 호 길이 0인 점 곡선이면 논문의 선적분 정의상 경로 길이도 0이어서
  결과가 0입니다.
- 이 구현의 `CDTW`는 단순히 DTW 행렬을 보간한 것이 아니라, 호 길이로 매개화된
  연속 곡선의 L1 매개공간 선적분 정의를 근사합니다.

## 기준 논문

Kevin Buchin, Andre Nusser, and Sampson Wong, *Computing Continuous Dynamic
Time Warping of Time Series in Polynomial Time*, SoCG 2022,
DOI: 10.4230/LIPIcs.SoCG.2022.22.
