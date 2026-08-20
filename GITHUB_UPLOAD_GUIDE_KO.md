# GitHub 업로드 및 설치 가이드

이 폴더는 그대로 하나의 GitHub 저장소로 사용할 수 있습니다. 아래 명령에서
저장소 주소는 `sxhx2009-crypto/cdtw-python` 기준입니다.

## 1. GitHub에 빈 저장소 만들기

1. GitHub에서 **New repository**를 선택합니다.
2. 저장소 이름을 `cdtw-python`으로 지정합니다.
3. 공개하려면 Public, 개인적으로만 쓰려면 Private을 선택합니다.
4. README, `.gitignore`, 라이선스 자동 생성을 선택하지 않고 빈 저장소로 만듭니다.

## 2. Windows에서 파일 올리기

ZIP을 푼 `cdtw-python` 폴더에서 Git Bash 또는 VS Code 터미널을 열고 실행합니다.

```bash
git init -b main
git add .
git commit -m "Initial CDTW implementation"
git remote add origin https://github.com/sxhx2009-crypto/cdtw-python.git
git push -u origin main
```

Git이 이름과 이메일을 요구하면 한 번만 설정합니다.

```bash
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_EMAIL"
```

GitHub CLI가 설치되어 있다면 빈 저장소를 웹에서 먼저 만들지 않고 다음처럼
생성·업로드할 수도 있습니다.

```bash
git init -b main
git add .
git commit -m "Initial CDTW implementation"
gh auth login
gh repo create cdtw-python --public --source=. --remote=origin --push
```

비공개 저장소가 필요하면 `--public`을 `--private`으로 바꿉니다.

## 3. GitHub 저장소에서 바로 설치하기

공개 저장소라면 다른 컴퓨터에서 다음 명령만으로 설치할 수 있습니다.

```bash
py -m pip install "git+https://github.com/sxhx2009-crypto/cdtw-python.git@main"
```

macOS/Linux에서는 `py` 대신 `python3`를 사용할 수 있습니다. 설치 후:

```python
from cdtw import cdtw_distance

distance = cdtw_distance([0.0, 1.0, 0.0], [0.0, 0.8, 0.0])
print(distance)
```

## 4. 특정 검증 버전을 고정해서 설치하기

결과 재현을 위해 릴리스 태그를 만드는 것이 좋습니다.

```bash
git tag -a v0.2.1 -m "Validated CDTW v0.2.1"
git push origin v0.2.1
```

이 버전을 고정 설치하려면:

```bash
py -m pip install "git+https://github.com/sxhx2009-crypto/cdtw-python.git@v0.2.1"
```

연구 코드에서는 바뀔 수 있는 `main`보다 태그 또는 전체 커밋 해시를 고정하는
것이 재현성에 유리합니다.

## 5. 로컬 개발과 테스트

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -e ".[dev]"
py -m unittest discover -s tests -v
python validation/validation_suite.py
```

GitHub에 push하거나 pull request를 만들면 `.github/workflows/tests.yml`이
Python 3.10~3.13에서 단위 테스트와 패키지 빌드를 자동 실행합니다. 전체 전문
검증은 GitHub의 **Actions → tests → Run workflow**에서 수동 실행할 수 있습니다.

## 6. 수정 내용을 다시 올리기

```bash
git add .
git commit -m "Describe the change"
git push
```

## 7. 공개 전 라이선스 선택

현재 패키지는 사용자가 권리 조건을 직접 결정할 수 있도록 `LICENSE`를 넣지
않았습니다. 다른 사람이 복제·수정·배포하게 하려면 공개 전에 GitHub의
**Add file → Create new file → Choose a license template**에서 목적에 맞는
라이선스를 선택하세요. 라이선스가 없으면 일반적으로 저작권자가 모든 권리를
보유한 상태로 취급됩니다.
