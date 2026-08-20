# CDTW Boundary-Sampling for Python

[![tests](https://github.com/sxhx2009-crypto/cdtw-python/actions/workflows/tests.yml/badge.svg)](https://github.com/sxhx2009-crypto/cdtw-python/actions/workflows/tests.yml)

두 수열을 **1차원 다각선 곡선**으로 해석하고, 호 길이(arc length)로
매개화한 뒤 Continuous Dynamic Time Warping을 계산하는 NumPy 구현입니다.

이 저장소는 `pyproject.toml` 기반으로 설치할 수 있으며, push와 pull request마다
GitHub Actions가 Python 3.10~3.14 단위 테스트와, 예제·검증 스위트·패키지 빌드를
자동 검사합니다.

## GitHub에서 바로 설치

```bash
py -m pip install "git+https://github.com/sxhx2009-crypto/cdtw-python.git@v0.2.4"
```

최신 개발본이 필요하면 `@v0.2.4` 대신 `@main`을 사용합니다.
자세한 최초 업로드 방법은 [`GITHUB_UPLOAD_GUIDE_KO.md`](GITHUB_UPLOAD_GUIDE_KO.md)에
정리되어 있습니다.

## 먼저 읽으세요: 시간축을 쓰지 않습니다

**이 구현은 샘플 인덱스나 측정 시각을 좌표로 전혀 사용하지 않습니다.** 논문의
1차원 CDTW 정의대로, 값들을 잇는 곡선을 **값 공간의 호 길이**로 다시 매개화해
값의 궤적만 비교합니다. `(시각, 값)` 2차원 곡선을 기대한다면 이 구현은 원하는
것이 아닙니다.

여기서 직관에 반하는 결과 두 가지가 나옵니다.

```python
cdtw_distance([3.0, 3.0, 3.0], [5.0, 5.0])   # -> 0.0
cdtw_distance([2.0], [7.0])                   # -> 0.0
```

**상수 계열끼리는 값이 달라도 거리가 0입니다.** 두 곡선 모두 호 길이가 0인
점으로 줄어들어 적분 경로 자체의 길이가 0이기 때문입니다. 논문 정의상 올바른
값이지만, 평평한 신호를 다루는 실전 코드에서는 사고를 부르는 성질이므로
반드시 알고 쓰세요. 연속 중복값과 원래 표본 간격 정보도 같은 이유로 사라집니다.

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
조각별 이차 경계함수 전파를 사용하는 정확한 `O(n^5)` 알고리즘은 아닙니다.
`cdtw_adaptive`가 반환하는 `estimated_error`도 연속 최적값에 대한 보증된 오차
상한이 아니라, 최근 안정화 창에서 관측된 해상도 간 변화량의 최댓값입니다.

## 필요 패키지

```bash
python -m pip install numpy
```

저장소를 내려받아 수정하면서 사용하려면 다음처럼 editable 설치합니다.

```bash
git clone https://github.com/sxhx2009-crypto/cdtw-python.git
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

**평탄 구간 주의.** 값이 여러 해상도에 걸쳐 완전히 같다가 다시 떨어지는 일이
실제로 있습니다. 두 가지 보호 장치를 둡니다. 격자 크기를 2배로 해도
`grid_shape`가 그대로면 값이 같은 것이 당연하므로 증거로 세지 않고, 창 전체가
동일하면 `estimated_error`를 0이 아니라 **마지막으로 값이 움직인 폭**으로
보고합니다. 그래도 무작위 120쌍 중 1쌍은 여전히 평탄 구간 뒤에 감소가
이어졌습니다. `grid_size`가 작을 때의 `converged=True`는 약한 증거로 보세요.

Richardson 외삽은 **의도적으로 쓰지 않습니다.** 관측된 수렴 차수는 `[0,1,0,1]`
대 `[0,1]` 같은 단순 입력에서만 정확히 2이고, 실제 곡선에서는 30.6, -33.6,
0.35처럼 불안정합니다. 게다가 평탄 구간에 적용하면 `estimated_error`를 정확히
0으로 보고해, 막으려던 문제를 오히려 되살립니다. 또한 반환되는 `distance`는
항상 **실제로 실현 가능한 단조 경로의 비용**(따라서 참 최적값의 상계)인데,
외삽값은 그 성질을 잃습니다.

논문의 원래 선적분값이 기본값입니다. 서로 다른 곡선 길이의 영향을 줄인 평균
비용이 필요하면 `normalized=True`를 지정할 수 있습니다.

## 테스트 실행

저장소 루트에서 다음 명령을 실행합니다.

```bash
python -m unittest discover -s tests -v
python examples/basic_usage.py
python validation/validation_suite.py
```

29개 단위 테스트에는 동일 곡선의 거리 0, 대칭성, 단일 셀과 다중 셀 각각의
닫힌형 해, 퇴화한 점-선분, 격자 세분 단조성, 독립 격자-DP 상계, 공선점
재표본화의 정확한 불변성, 극단적 크기비와 미소 곡선, overflow 예외,
`cdtw_adaptive`의 경로 비용 일치·평탄 구간·인자 검증이 포함됩니다.
`validation_suite.py`는
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
.gitignore                      빌드·캐시 산출물 제외 규칙
LICENSE                         MIT 라이선스 전문
```

## 라이선스

MIT License. 전문은 [`LICENSE`](LICENSE)에 있습니다.
저작권자는 두 기여자 `sj10132`, `sxhx2009-crypto` 입니다.
복제·수정·배포·상업적 사용이 모두 허용되며, 저작권 고지와 라이선스 전문을
함께 포함하기만 하면 됩니다.

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
- 같은 방향으로 이어지는 연속 선분은 하나로 병합되어 **꼭짓점(turning point)만
  남습니다.** `[0, 1, 2]`는 `[0, 2]`와 완전히 동일하게 처리되므로 공선점을
  끼워 넣는 재표본화는 **정확히 불변**입니다(무작위 200쌍 상대편차 `0.0`).
  셀 수가 줄어드는 부수 효과로 속도도 빨라집니다.
- `grid_size`는 **선분당이 아니라 긴 곡선 전체의 분할 수**입니다. 따라서 꼭짓점이
  많은 계열일수록 셀 하나에 들어가는 정규 표본이 줄어듭니다. `grid_size=256`
  기준 실측으로 `n=50`이면 셀당 평균 10.1개지만 `n=200`이면 2.3개이고 셀의 22%는
  내부 표본이 0개(코너로만 통과 가능)였습니다. 긴 계열에서는 `grid_size`를
  꼭짓점 수에 비례해 키우거나 `cdtw_adaptive`를 쓰세요.
- 입력 크기에는 상한이 있습니다. 개별 값뿐 아니라 **두 곡선의 조합**을 검사합니다.
  최대 높이 `H`와 총 호 길이 `L`에 대해 `2·H²`(셀 내부 제곱항)와 `L·H`(누적 적분)이
  float64 범위를 넘으면 `ValueError`를 냅니다. 값 각각은 표현 가능해도 누적합이
  넘칠 수 있어서, 한쪽만 검사하면 `inf`와 `RuntimeWarning`이 새어 나옵니다.
  상계라서 실제로 계산 가능한 입력(예: `5e299`가 나오는 조합)은 막지 않습니다.
- **노이즈에 민감합니다.** 호 길이는 곧 **총변동(total variation)**이라, 값이
  아주 조금만 되돌아가도 꼭짓점이 하나 늘고 셀이 늘어납니다.

  ```python
  [0, 1, 1 + 1e-16, 2]   # -> 꼭짓점 2개 (같은 방향이라 병합)
  [0, 1, 1 - 1e-16, 2]   # -> 꼭짓점 4개, 셀 3개  (1e-16 반전이 꼭짓점이 됨)
  ```

  정의상 올바른 동작이지만, 측정 노이즈가 있는 신호는 셀 수와 실행시간이 크게
  불어납니다. 필요하면 미리 스무딩하거나 진폭이 임계값보다 작은 반전을 제거하세요.
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
