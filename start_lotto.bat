@echo off
echo Starting Lotto App...

:: C:\lotto-app 폴더로 이동
C:
cd C:\lotto-app

:: 가상 환경 활성화
call .\venv\Scripts\activate

:: 스트림릿 앱 실행
streamlit run app.py