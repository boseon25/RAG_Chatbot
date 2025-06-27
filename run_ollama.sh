#!/bin/bash

echo "Ollama 설치 중..."
curl -fsSL https://ollama.com/install.sh | sh

echo "설치 확인:"
ollama --version

echo "Ollama 서버 실행 중 (백그라운드)..."
nohup ollama serve > log.txt 2>&1 &

# 서버 시작 대기 시간 (선택적으로 3초 정도 기다려줌)
sleep 3

echo "모델 다운로드: exaone3.5:2.4b"
ollama pull exaone3.5:2.4b

echo "모델 상태 확인:"
curl http://localhost:11434/api/tags

echo "모든 작업 완료!"