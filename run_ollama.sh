#!/bin/bash

echo "Ollama ��ġ ��..."
curl -fsSL https://ollama.com/install.sh | sh

echo "��ġ Ȯ��:"
ollama --version

echo "Ollama ���� ���� �� (��׶���)..."
nohup ollama serve > log.txt 2>&1 &

# ���� ���� ��� �ð� (���������� 3�� ���� ��ٷ���)
sleep 3

echo "�� �ٿ�ε�: exaone3.5:2.4b"
ollama pull exaone3.5:2.4b

echo "�� ���� Ȯ��:"
curl http://localhost:11434/api/tags

echo "��� �۾� �Ϸ�!"